"""`lite configure claude` and `lite unconfigure claude`: persistent Claude Code wiring, undoable."""

import re
import sys
from collections.abc import Callable, Sequence
from typing import Final

import click
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from .auth import CliContextObj, context_secret_vault, get_stored_api_key
from .claude_settings import (
    CLAUDE_SETTINGS_PATH,
    CONFIGURE_STATE_PATH,
    SETTINGS_FILE_OWNERS,
    ApiKeyHelper,
    ClaudeCredential,
    ClaudeSettingsError,
    ModelChoice,
    StartOn,
    StaticToken,
    UnpinModel,
    configure_claude_settings,
    resolve_api_key_helper,
    unconfigure_claude_settings,
)
from .pi import PiSyncError, fetch_model_ids
from .up import ensure_fresh_login

_LISTED_MODELS_SHOWN: Final = 20
_CLAUDE_TARGET: Final = "claude"
_TARGETS: Final = ((_CLAUDE_TARGET, "Claude Code (CLI)"),)
_KEEP_DEFAULT_MODEL: Final = "Keep Claude Code's own default"
_CLAUDE_CODE_PICKER_FILTER: Final = re.compile(r"claude|anthropic", re.IGNORECASE)
_REJECTED_STATUSES: Final = frozenset((401, 403))


def resolve_credential(ctx: click.Context, api_key: str | None) -> tuple[ClaudeCredential, str]:
    """The credential to write and the key to check the proxy with.

    An explicit key (--api-key, `lite --api-key`, LITELLM_PROXY_API_KEY) is long-lived and goes
    into settings.json as a static token. Without one, the stored `lite login` credential is used
    the way `lite login --config-claude` uses it, through apiKeyHelper, since it expires within a
    day and renews in place there; a missing or stale login is refreshed first, as `lite up` does.
    """
    ctx_obj: Final[CliContextObj] = ctx.obj
    explicit: Final = api_key or (None if ctx_obj.get("api_key_from_token_file") else ctx_obj.get("api_key"))
    if explicit:
        return StaticToken(explicit), explicit
    base_url: Final = ctx_obj["base_url"]
    ensure_fresh_login(ctx)
    stored: Final = get_stored_api_key(expected_base_url=base_url, vault=context_secret_vault(ctx))
    if not stored:
        raise ClaudeSettingsError("Login did not produce a usable token.")
    return ApiKeyHelper(resolve_api_key_helper(base_url)), stored


def _listing_error(base_url: str, error: PiSyncError) -> str:
    if error.status in _REJECTED_STATUSES:
        return f"LiteLLM rejected your key (HTTP {error.status}). Run `lite login` to refresh it, or pass a valid --api-key."
    return f"{error.message} Is the proxy at {base_url} running, and is --base-url (or LITELLM_PROXY_URL) correct?"


def _listed_models(base_url: str, key: str) -> tuple[str, ...]:
    listed: Final = fetch_model_ids(base_url, key)
    if isinstance(listed, PiSyncError):
        raise click.ClickException(_listing_error(base_url, listed))
    return listed


def _model_choice(model: str | None) -> ModelChoice:
    return StartOn(model) if model is not None else UnpinModel()


def _apply_claude(ctx: click.Context, credential: ClaudeCredential, listed: Sequence[str], model: str | None) -> None:
    ctx_obj: Final[CliContextObj] = ctx.obj
    base_url: Final = ctx_obj["base_url"]
    if model is not None and model not in listed:
        shown: Final = ", ".join(listed[:_LISTED_MODELS_SHOWN])
        more: Final = f", and {len(listed) - _LISTED_MODELS_SHOWN} more" if len(listed) > _LISTED_MODELS_SHOWN else ""
        raise click.ClickException(
            f"{model!r} is not served by {base_url} for this key. /v1/models lists: {shown}{more}."
        )
    try:
        configure_claude_settings(
            base_url, credential, _model_choice(model), CLAUDE_SETTINGS_PATH, CONFIGURE_STATE_PATH, SETTINGS_FILE_OWNERS
        )
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    in_picker: Final = sum(1 for listed_model in listed if _CLAUDE_CODE_PICKER_FILTER.search(listed_model))
    click.echo(f"Configured Claude Code: {CLAUDE_SETTINGS_PATH} now routes through {base_url}.")
    click.echo(
        "Credential: your virtual key, stored in the file as ANTHROPIC_AUTH_TOKEN."
        if isinstance(credential, StaticToken)
        else "Credential: your `lite login`, read through apiKeyHelper on every request, so a later login renews it."
    )
    click.echo(
        f"Starting model: {model}; switch any time with /model."
        if model is not None
        else "Starting model: not pinned (Claude Code's default, or a model you set yourself); switch with /model, or "
        "pass --model to start on a proxy model."
    )
    click.echo(
        f"/model will list {in_picker} of the proxy's {len(listed)} models (Claude Code shows only ids containing "
        "'claude' or 'anthropic')."
    )
    click.echo("Start `claude` from any terminal. Undo with `lite unconfigure claude`.")
    if isinstance(credential, StaticToken) and CLAUDE_SETTINGS_PATH.is_symlink():
        click.echo(
            f"Note: {CLAUDE_SETTINGS_PATH} is a symlink to {CLAUDE_SETTINGS_PATH.resolve()}, so your key now lives in "
            "that file; keep it out of version control.",
            err=True,
        )


def _pick_targets() -> tuple[str, ...]:
    picked: Final = inquirer.checkbox(
        message="Which agents should route through LiteLLM?",
        choices=[Choice(value, name=label, enabled=True) for value, label in _TARGETS],
        validate=lambda chosen: len(chosen) > 0,
        invalid_message="Pick at least one.",
    ).execute()
    return tuple(str(value) for value in picked)


def _pick_model(listed: Sequence[str]) -> str | None:
    picked: Final = inquirer.fuzzy(
        message="Model Claude Code starts on (type to filter; /model switches any time):",
        choices=[_KEEP_DEFAULT_MODEL, *listed],
    ).execute()
    return None if picked == _KEEP_DEFAULT_MODEL else str(picked)


def interactive_configure(
    ctx: click.Context,
    pick_targets: Callable[[], tuple[str, ...]] = _pick_targets,
    pick_model: Callable[[Sequence[str]], str | None] = _pick_model,
) -> None:
    """`lite configure` with no agent named: ask which agents to wire and which model to pin."""
    targets: Final = pick_targets()
    if _CLAUDE_TARGET not in targets:
        return
    ctx_obj: Final[CliContextObj] = ctx.obj
    try:
        credential, key = resolve_credential(ctx, None)
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    listed: Final = _listed_models(ctx_obj["base_url"], key)
    _apply_claude(ctx, credential, listed, pick_model(listed))


@click.group(name="configure", invoke_without_command=True)
@click.pass_context
def configure_group(ctx: click.Context) -> None:
    """Persistently route a coding agent through your LiteLLM proxy.

    With no agent named, asks which agents to wire and which proxy model to pin.
    """
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty():
        raise click.ClickException(
            "`lite configure` asks questions, so it needs a terminal. Non-interactively, run "
            "`lite configure claude --api-key <key> --model <model>`."
        )
    interactive_configure(ctx)


@click.group(name="unconfigure")
def unconfigure_group() -> None:
    """Undo `lite configure` for a coding agent."""


@configure_group.command(name="claude")
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help="Long-lived LiteLLM virtual key written into Claude Code's settings. Defaults to the `lite --api-key` / "
    "LITELLM_PROXY_API_KEY value; with neither, your `lite login` credential is used through apiKeyHelper.",
)
@click.option(
    "--model",
    default=None,
    help="Proxy model Claude Code starts on (the /model picker's default row). Must be listed on /v1/models for "
    "the key; without it Claude Code keeps its own default.",
)
@click.pass_context
def configure_claude(ctx: click.Context, api_key: str | None, model: str | None) -> None:
    """Route every Claude Code session through your LiteLLM proxy until `lite unconfigure claude`.

    Patches ~/.claude/settings.json in place: the proxy URL, your credential (a virtual key as a
    static token, or your `lite login` through apiKeyHelper), and gateway model discovery so
    /model lists the proxy's models; --model picks the one Claude Code starts on. Every other
    setting is kept, and what changed is recorded so `lite unconfigure claude` can put it back.
    Assumes the proxy is already running.
    """
    ctx_obj: Final[CliContextObj] = ctx.obj
    try:
        credential, key = resolve_credential(ctx, api_key)
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    _apply_claude(ctx, credential, _listed_models(ctx_obj["base_url"], key), model)


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
    click.echo(
        f"Restored {CLAUDE_SETTINGS_PATH}: {', '.join(outcome.restored) or 'nothing was still ours to restore'}."
    )
    if outcome.kept:
        click.echo(f"Left as you changed them since: {', '.join(outcome.kept)}.")
    if outcome.withheld:
        click.echo(
            f"Left removed, since env.ANTHROPIC_BASE_URL was changed after configure and they belong to the old "
            f"endpoint: {', '.join(outcome.withheld)}. Put them back by hand if that server should have them."
        )


__all__ = ("configure_group", "interactive_configure", "resolve_credential", "unconfigure_group")
