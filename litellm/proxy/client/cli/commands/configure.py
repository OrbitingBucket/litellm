"""`lite configure [claude|codex]` and `lite unconfigure [claude|codex]`: persistent agent wiring, undoable."""

import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import click
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from .agent_config import ROOT_SECTION, fingerprint, read_configure_receipt
from .agents import is_interactive, launch_agent
from .auth import CliContextObj, context_secret_vault, get_stored_api_key
from .claude_settings import (
    CLAUDE_SETTINGS_PATH,
    CONFIGURE_STATE_PATH,
    MODEL_KEY,
    SETTINGS_FILE_OWNERS,
    ApiKeyHelper,
    ClaudeCredential,
    ClaudeSettingsError,
    ModelChoice,
    StartOn,
    StaticToken,
    UnconfigureOutcome,
    UnpinModel,
    configure_claude_settings,
    print_token_command,
    resolve_api_key_helper,
    unconfigure_claude_settings,
)
from .codex_settings import (
    CODEX_CONFIG_PATH,
    CODEX_CONFIGURE_STATE_PATH,
    configure_codex_config,
    unconfigure_codex_config,
)
from .pi import PiSyncError, fetch_model_ids, fetch_model_limits
from .up import ensure_fresh_login

_LISTED_MODELS_SHOWN: Final = 20
_CLAUDE_CODE_PICKER_FILTER: Final = re.compile(r"claude|anthropic", re.IGNORECASE)
_REJECTED_STATUSES: Final = frozenset((401, 403))
CLAUDE_TARGET: Final = "claude"
CODEX_TARGET: Final = "codex"
TARGETS: Final = ((CLAUDE_TARGET, "Claude Code (CLI)"), (CODEX_TARGET, "Codex (CLI and app)"))
_TARGET_LABELS: Final[Mapping[str, str]] = MappingProxyType(dict(TARGETS))
_TOKEN_READERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        CLAUDE_TARGET: "apiKeyHelper reads this token on every Claude Code request",
        CODEX_TARGET: "Codex's provider auth command reads this token on every request",
    }
)
_KEEP_DEFAULT_MODEL: Final = "Keep the agent's own default"


@dataclass(frozen=True, slots=True)
class Session:
    """What every configure step shares: the proxy, the credential, the key it was checked with, the models it
    serves, and whether stdin was a tty before anything could run a browser login."""

    base_url: str
    credential: ClaudeCredential
    key: str
    listed: tuple[str, ...]
    started_interactive: bool

    def launch(self, agent: str) -> None:
        """Hand off to the agent; the key was already checked against /v1/models, so skip the launcher's probe."""
        launch_agent(self.base_url, self.key, agent, skip_verify=True, started_interactive=self.started_interactive)


def resolve_credential(
    ctx: click.Context, api_key: str | None, agent: str = CLAUDE_TARGET
) -> tuple[ClaudeCredential, str]:
    """The credential to write and the key to check the proxy with.

    An explicit key (--api-key, `lite --api-key`, LITELLM_PROXY_API_KEY) is long-lived and goes
    into the agent's config as a static token. Without one, the stored `lite login` credential is
    used the way `lite login --config-claude` uses it, through a `lite auth print-token` command
    the agent runs, since it expires within a day and renews in place there; a missing or stale
    login is refreshed first, as `lite up` does.
    """
    ctx_obj: Final[CliContextObj] = ctx.obj
    explicit: Final = api_key or (None if ctx_obj.get("api_key_from_token_file") else ctx_obj.get("api_key"))
    if explicit:
        return StaticToken(explicit), explicit
    base_url: Final = ctx_obj["base_url"]
    ensure_fresh_login(ctx, reader=_TOKEN_READERS[agent])
    stored: Final = get_stored_api_key(expected_base_url=base_url, vault=context_secret_vault(ctx))
    if not stored:
        raise ClaudeSettingsError("Login did not produce a usable token.")
    return ApiKeyHelper(resolve_api_key_helper(base_url)), stored


def _listing_error(base_url: str, error: PiSyncError) -> str:
    if error.status in _REJECTED_STATUSES:
        return f"LiteLLM rejected your key (HTTP {error.status}). Run `lite login` to refresh it, or pass a valid --api-key."
    return f"{error.message} Is the proxy at {base_url} running, and is --base-url (or LITELLM_PROXY_URL) correct?"


def open_session(ctx: click.Context, api_key: str | None, agent: str = CLAUDE_TARGET) -> Session:
    ctx_obj: Final[CliContextObj] = ctx.obj
    base_url: Final = ctx_obj["base_url"]
    started_interactive: Final = is_interactive()
    try:
        credential, key = resolve_credential(ctx, api_key, agent)
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    listed: Final = fetch_model_ids(base_url, key)
    if isinstance(listed, PiSyncError):
        raise click.ClickException(_listing_error(base_url, listed))
    return Session(base_url, credential, key, listed, started_interactive)


def _require_listed(session: Session, model: str | None) -> ModelChoice:
    if model is None:
        return UnpinModel()
    if model not in session.listed:
        shown: Final = ", ".join(session.listed[:_LISTED_MODELS_SHOWN])
        more: Final = (
            f", and {len(session.listed) - _LISTED_MODELS_SHOWN} more"
            if len(session.listed) > _LISTED_MODELS_SHOWN
            else ""
        )
        raise click.ClickException(
            f"{model!r} is not served by {session.base_url} for this key. /v1/models lists: {shown}{more}."
        )
    return StartOn(model)


def _credential_line(session: Session, agent: str) -> str:
    if isinstance(session.credential, StaticToken):
        return "Credential: your virtual key, stored in the file as a bearer token."
    runs: Final = "Claude Code" if agent == CLAUDE_TARGET else "Codex"
    return f"Credential: your `lite login`, which {runs} reads through `lite auth print-token` on demand, so a later login renews it."


def _model_line(label: str, agent_default: str, model: str | None, released: bool, detail: str = "") -> str:
    if model is not None:
        return f"{label}: {model}{detail}."
    if released:
        return f"{label}: the pin an earlier configure made is released, back to {agent_default}."
    return f"{label}: not pinned ({agent_default})."


def _tracked_note(path: Path) -> None:
    """Say so when the file or its directory is a symlink, the shape a dotfiles repository takes."""
    if not (path.is_symlink() or path.parent.is_symlink()):
        return
    click.echo(
        f"Note: {path} resolves to {path.resolve()}, so your key now lives in that file; keep it out of version control.",
        err=True,
    )


def apply_claude(session: Session, model: str | None) -> None:
    choice: Final = _require_listed(session, model)
    released: Final = _pin_released(CONFIGURE_STATE_PATH, choice)
    try:
        configure_claude_settings(
            session.base_url,
            session.credential,
            choice,
            CLAUDE_SETTINGS_PATH,
            CONFIGURE_STATE_PATH,
            SETTINGS_FILE_OWNERS,
        )
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    in_picker: Final = sum(1 for listed_model in session.listed if _CLAUDE_CODE_PICKER_FILTER.search(listed_model))
    click.echo(f"Configured Claude Code: {CLAUDE_SETTINGS_PATH} now routes through {session.base_url}.")
    click.echo(_credential_line(session, CLAUDE_TARGET))
    click.echo(
        _model_line(
            "Starting model",
            "Claude Code's default, or a model you set yourself; switch with /model, or pass --model to start on a "
            "proxy model",
            model,
            released,
            "; switch any time with /model",
        )
    )
    click.echo(
        f"/model will list {in_picker} of the proxy's {len(session.listed)} models (Claude Code shows only ids containing "
        "'claude' or 'anthropic')."
    )
    click.echo("Start `claude` from any terminal. Undo with `lite unconfigure claude`.")
    if isinstance(session.credential, StaticToken):
        _tracked_note(CLAUDE_SETTINGS_PATH)


def apply_codex(session: Session, model: str | None) -> None:
    choice: Final = _require_listed(session, model)
    released: Final = _pin_released(CODEX_CONFIGURE_STATE_PATH, choice)
    limits: Final = fetch_model_limits(session.base_url, session.key) if model is not None else MappingProxyType({})
    context_window: Final = limits[model].context_window if model is not None and model in limits else None
    try:
        configure_codex_config(
            session.base_url,
            session.credential,
            lambda: print_token_command(session.base_url),
            choice,
            context_window,
            CODEX_CONFIG_PATH,
            CODEX_CONFIGURE_STATE_PATH,
        )
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    click.echo(f"Configured Codex: {CODEX_CONFIG_PATH} now routes through {session.base_url}.")
    click.echo(_credential_line(session, CODEX_TARGET))
    window: Final = f", context window {context_window} tokens from the proxy" if context_window else ""
    click.echo(
        _model_line(
            "Model",
            "Codex's own default id, which then has to exist on the proxy; pass --model to pick one it serves",
            model,
            released,
            window,
        )
    )
    click.echo(
        "Codex's /model picker still shows its built-in catalog; to switch to another proxy model, run "
        "`lite configure codex --model <id>` again or pass `codex -m <id>`."
    )
    click.echo("Start `codex` from any terminal or the Codex app. Undo with `lite unconfigure codex`.")
    if isinstance(session.credential, StaticToken):
        _tracked_note(CODEX_CONFIG_PATH)


def _pin_released(state_path: Path, choice: ModelChoice) -> bool:
    """Whether this run lets go of a model an earlier configure pinned, so the receipt line can say so."""
    if not isinstance(choice, UnpinModel):
        return False
    try:
        earlier: Final = read_configure_receipt(state_path)
    except ClaudeSettingsError:
        return False
    root: Final = earlier.sections.get(ROOT_SECTION) if earlier is not None else None
    return (
        root is not None
        and MODEL_KEY in root.written
        and root.written[MODEL_KEY] != fingerprint(root.previous[MODEL_KEY])
    )


_APPLY: Final[Mapping[str, Callable[[Session, str | None], None]]] = MappingProxyType(
    {CLAUDE_TARGET: apply_claude, CODEX_TARGET: apply_codex}
)


def _target_choices() -> list[Choice]:  # mutable-ok: InquirerPy takes a list
    return [Choice(v, name=n, enabled=v == CLAUDE_TARGET) for v, n in TARGETS]  # mutable-ok: InquirerPy list


def _pick_targets() -> tuple[str, ...]:
    picked: Final = inquirer.checkbox(
        message="Which agents should route through LiteLLM?",
        choices=_target_choices(),
        validate=lambda chosen: len(chosen) > 0,
        invalid_message="Pick at least one.",
    ).execute()
    return tuple(str(value) for value in picked)


def _pick_model(agent: str, listed: Sequence[str]) -> str | None:
    picked: Final = inquirer.fuzzy(
        message=f"Model {_TARGET_LABELS[agent]} starts on (type to filter):",
        choices=[_KEEP_DEFAULT_MODEL, *listed],  # mutable-ok: InquirerPy takes a list
    ).execute()
    return None if picked == _KEEP_DEFAULT_MODEL else str(picked)


def _confirm_launch(agent: str) -> bool:
    answer: Final[object] = inquirer.confirm(message=f"Start {_TARGET_LABELS[agent]} now?", default=True).execute()
    return answer is True


def interactive_configure(
    ctx: click.Context,
    pick_targets: Callable[[], tuple[str, ...]] = _pick_targets,
    pick_model: Callable[[str, Sequence[str]], str | None] = _pick_model,
    confirm_launch: Callable[[str], bool] = _confirm_launch,
    launch: Callable[[Session, str], None] = Session.launch,
) -> None:
    """`lite configure` with no agent named: ask which agents to wire, which model each starts on, and whether to start one."""
    targets: Final = tuple(target for target in pick_targets() if target in _APPLY)
    if not targets:
        return
    session: Final = open_session(ctx, None, targets[0])
    for target in targets:
        _APPLY[target](session, pick_model(target, session.listed))
    for target in targets:
        if confirm_launch(target):
            launch(session, target)
            return


@click.group(name="configure", invoke_without_command=True)
@click.pass_context
def configure_group(ctx: click.Context) -> None:
    """Persistently route a coding agent through your LiteLLM proxy.

    With no agent named, asks which agents to wire, which proxy model each starts on, and
    whether to start one right away.
    """
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty():
        raise click.ClickException(
            "`lite configure` asks questions, so it needs a terminal. Non-interactively, run "
            "`lite configure claude --api-key <key> --model <model>` or `lite configure codex ...`."
        )
    interactive_configure(ctx)


@click.group(name="unconfigure")
def unconfigure_group() -> None:
    """Undo `lite configure` for a coding agent."""


_API_KEY_HELP: Final = (
    "Long-lived LiteLLM virtual key written into the agent's config. Defaults to the `lite --api-key` / "
    "LITELLM_PROXY_API_KEY value; with neither, your `lite login` credential is used through `lite auth print-token`."
)
_LAUNCH_HELP: Final = "Start the agent right after configuring it."


@configure_group.command(name="claude")
@click.option("--api-key", "api_key", default=None, help=_API_KEY_HELP)
@click.option(
    "--model",
    default=None,
    help="Proxy model Claude Code starts on (the /model picker's default row). Must be listed on /v1/models for "
    "the key; without it Claude Code keeps its own default.",
)
@click.option("--launch", is_flag=True, default=False, help=_LAUNCH_HELP)
@click.pass_context
def configure_claude(ctx: click.Context, api_key: str | None, model: str | None, launch: bool) -> None:
    """Route every Claude Code session through your LiteLLM proxy until `lite unconfigure claude`.

    Patches ~/.claude/settings.json (or $CLAUDE_CONFIG_DIR/settings.json) in place: the proxy URL,
    your credential (a virtual key as a static token, or your `lite login` through apiKeyHelper),
    and gateway model discovery so /model lists the proxy's models; --model picks the one Claude
    Code starts on. Every other setting is kept, and what changed is recorded so
    `lite unconfigure claude` can put it back. Assumes the proxy is already running.
    """
    session: Final = open_session(ctx, api_key, CLAUDE_TARGET)
    apply_claude(session, model)
    if launch:
        session.launch(CLAUDE_TARGET)


@configure_group.command(name="codex")
@click.option("--api-key", "api_key", default=None, help=_API_KEY_HELP)
@click.option(
    "--model",
    default=None,
    help="Proxy model Codex uses. Must be listed on /v1/models for the key; without it Codex keeps its own default id.",
)
@click.option("--launch", is_flag=True, default=False, help=_LAUNCH_HELP)
@click.pass_context
def configure_codex(ctx: click.Context, api_key: str | None, model: str | None, launch: bool) -> None:
    """Route Codex, the CLI and the desktop app, through your LiteLLM proxy until `lite unconfigure codex`.

    Patches ~/.codex/config.toml (or $CODEX_HOME/config.toml) in place: a `litellm` model
    provider on the Responses transport, `model_provider` pointing at it, and, with --model,
    that model plus its context window from the proxy. Comments and every other key are kept,
    and what changed is recorded so `lite unconfigure codex` can put it back.
    """
    session: Final = open_session(ctx, api_key, CODEX_TARGET)
    apply_codex(session, model)
    if launch:
        session.launch(CODEX_TARGET)


@unconfigure_group.command(name="claude")
def unconfigure_claude() -> None:
    """Return Claude Code's settings to what they were before `lite configure claude`.

    Also undoes `lite login --config-claude`. Only keys still holding what configure wrote are
    put back; anything you changed since is left as it is and named in the output.
    """
    try:
        outcome: Final = unconfigure_claude_settings(CLAUDE_SETTINGS_PATH, CONFIGURE_STATE_PATH, SETTINGS_FILE_OWNERS)
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    _report_unconfigure(CLAUDE_SETTINGS_PATH, outcome)


@unconfigure_group.command(name="codex")
def unconfigure_codex() -> None:
    """Return Codex's config.toml to what it was before `lite configure codex`.

    Only keys still holding what configure wrote are put back; anything you changed since is
    left as it is and named in the output.
    """
    try:
        outcome: Final = unconfigure_codex_config(CODEX_CONFIG_PATH, CODEX_CONFIGURE_STATE_PATH)
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    _report_unconfigure(CODEX_CONFIG_PATH, outcome)


def _report_unconfigure(path: Path, outcome: UnconfigureOutcome) -> None:
    click.echo(f"Restored {path}: {', '.join(outcome.restored) or 'nothing was still ours to restore'}.")
    if outcome.kept:
        click.echo(f"Left as you changed them since: {', '.join(outcome.kept)}.")
    if outcome.withheld:
        click.echo(
            "Left removed, since the endpoint they belong to was changed after configure: "
            f"{', '.join(outcome.withheld)}. Put them back by hand if that server should have them."
        )


__all__ = (
    "Session",
    "configure_group",
    "interactive_configure",
    "open_session",
    "resolve_credential",
    "unconfigure_group",
)
