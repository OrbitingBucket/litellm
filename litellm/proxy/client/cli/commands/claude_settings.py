"""Shared handling of Claude Code's ~/.claude/settings.json.

`lite up` and `lite autoroute up` patch this file temporarily and restore it on
exit; `lite login --config-claude` and `lite configure claude` patch it
persistently and record how to undo it. All of them need the same merge and the
same apiKeyHelper command, and `up` already imports from `auth`, so the shared
parts live here rather than in any one command module.
"""

import hashlib
import json
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.private_json import (
    commit_staged_json,
    discard_staged_json,
    ensure_private_dir,
    stage_private_json,
)

from .cmd_quoting import quote_for_cmd

ENV_KEY: Final = "env"
API_KEY_HELPER_KEY: Final = "apiKeyHelper"
MODEL_KEY: Final = "model"
ANTHROPIC_BASE_URL_KEY: Final = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_KEY: Final = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_KEY: Final = "ANTHROPIC_API_KEY"
ENABLE_TOOL_SEARCH_KEY: Final = "ENABLE_TOOL_SEARCH"
ENABLE_TOOL_SEARCH_VALUE: Final = "true"
ENABLE_GATEWAY_MODEL_DISCOVERY_KEY: Final = "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"
ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE: Final = "1"
ANTHROPIC_DEFAULT_MODEL_ENV_KEYS: Final = (
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
)
OWNED_ENV_KEYS: Final = (
    ENABLE_TOOL_SEARCH_KEY,
    ENABLE_GATEWAY_MODEL_DISCOVERY_KEY,
    ANTHROPIC_BASE_URL_KEY,
    ANTHROPIC_AUTH_TOKEN_KEY,
    ANTHROPIC_API_KEY_KEY,
)
OWNED_TOP_LEVEL_KEYS: Final = (API_KEY_HELPER_KEY, MODEL_KEY)
_CREDENTIAL_ENV_KEYS: Final = frozenset((ANTHROPIC_API_KEY_KEY, ANTHROPIC_AUTH_TOKEN_KEY))

CLAUDE_SETTINGS_PATH: Final = Path.home() / ".claude" / "settings.json"
BACKUP_PATH: Final = Path.home() / ".litellm" / "claude_settings_backup.json"
AUTOROUTE_BACKUP_PATH: Final = Path.home() / ".litellm" / "autorouter" / "claude_settings_backup.json"
CONFIGURE_STATE_PATH: Final = Path.home() / ".litellm" / "claude_configure_state.json"


@dataclass(frozen=True, slots=True)
class SettingsFileOwner:
    """A command that takes temporary ownership of CLAUDE_SETTINGS_PATH and restores it later."""

    backup_path: Path
    start_command: str
    stop_command: str


SETTINGS_FILE_OWNERS: Final = (
    SettingsFileOwner(BACKUP_PATH, "lite up", "lite down"),
    SettingsFileOwner(AUTOROUTE_BACKUP_PATH, "lite autoroute up", "lite autoroute down"),
)

_SETTINGS_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


class ClaudeSettingsError(Exception):
    """Raised for any user-actionable failure while reading or writing Claude Code settings."""


@dataclass(frozen=True, slots=True)
class StaticToken:
    """A long-lived virtual key, written into env.ANTHROPIC_AUTH_TOKEN."""

    token: str


@dataclass(frozen=True, slots=True)
class ApiKeyHelper:
    """A `lite auth print-token` command Claude Code runs per request, so a login renews in place."""

    command: str


ClaudeCredential: TypeAlias = StaticToken | ApiKeyHelper


@dataclass(frozen=True, slots=True)
class KeepModel:
    """Leave the top-level `model` as it is, the user's or an earlier configure's (a re-login)."""


@dataclass(frozen=True, slots=True)
class UnpinModel:
    """Let go of a `model` an earlier configure pinned; one the user set themselves stays."""


@dataclass(frozen=True, slots=True)
class StartOn:
    """Pin the top-level `model`, the row Claude Code starts on."""

    model: str


ModelChoice: TypeAlias = KeepModel | UnpinModel | StartOn


class OwnedValue(BaseModel):
    """What one key held at a moment in time; `present=False` is an absent key, not a null one."""

    model_config = ConfigDict(frozen=True)

    present: bool
    value: JsonValue = None


class ConfigureReceipt(BaseModel):
    """What `lite configure claude` found and what it wrote, so unconfigure can undo only its own work.

    `previous_*` hold the values every owned key had before the first configure; a repeat
    configure keeps them, since the values it would otherwise snapshot are its own. `written_*`
    hold fingerprints of what was written, so unconfigure can tell a key it still owns from one
    the user changed since, without keeping a second copy of the token on disk. The file and
    `env` shapes are recorded separately from the keys, so a settings.json that did not exist,
    or an `env` that was absent or null, comes back exactly that way.
    """

    model_config = ConfigDict(frozen=True)

    file_existed: bool
    env_present: bool
    env_was_object: bool
    previous_env: Mapping[str, OwnedValue]
    previous_top_level: Mapping[str, OwnedValue]
    written_env: Mapping[str, str]
    written_top_level: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class UnconfigureOutcome:
    """Which owned keys unconfigure put back, which it left because the user had changed them, and which
    credentials it left removed because the endpoint they belonged to was changed after configure."""

    restored: tuple[str, ...]
    kept: tuple[str, ...]
    withheld: tuple[str, ...] = ()


def load_json_or_empty(path: Path) -> dict[str, JsonValue]:
    try:
        content: Final = path.read_bytes() if path.exists() else b""
    except OSError as e:
        raise ClaudeSettingsError(f"Could not read {path}: {e}") from e
    if not content.strip():
        return {}
    try:
        return _SETTINGS_ADAPTER.validate_json(content)
    except ValidationError:
        raise ClaudeSettingsError(
            f"{path} contains invalid JSON (or its root is not an object); cannot proceed safely."
        )


def _env_object(settings: Mapping[str, JsonValue], path: Path) -> Mapping[str, JsonValue]:
    raw_env: Final = settings.get(ENV_KEY)
    if raw_env is None:
        return MappingProxyType({})
    if not isinstance(raw_env, dict):
        raise ClaudeSettingsError(
            f'{path} has a non-object "{ENV_KEY}" value, which this would discard. Fix or remove it, then retry.'
        )
    return raw_env


def _refuse_while_owned(settings_path: Path, owners: Sequence[SettingsFileOwner]) -> None:
    for owner in owners:
        if owner.backup_path.exists():
            raise ClaudeSettingsError(
                f"`{owner.start_command}` is currently managing {settings_path} (backup at "
                f"{owner.backup_path}) and will restore it when it stops. "
                f"Run `{owner.stop_command}` first, then retry."
            )


def _write_target(settings_path: Path) -> Path:
    """Write through a symlinked settings.json rather than replacing the link.

    os.replace() would swap the symlink itself for a regular file, silently detaching a
    settings.json that is symlinked into a dotfiles repo, and there is no backup to undo that.
    """
    return settings_path.resolve() if settings_path.is_symlink() else settings_path


def _stage(path: Path, document: Mapping[str, object]) -> str:
    try:
        return stage_private_json(str(path), document)
    except OSError as e:
        raise ClaudeSettingsError(f"Could not write {path}: {e}") from e


def _write_settings(settings_path: Path, settings: Mapping[str, JsonValue]) -> None:
    target: Final = _write_target(settings_path)
    commit_staged_json(_stage(target, settings), str(target))


def merge_claude_settings(
    settings: Mapping[str, JsonValue],
    base_url: str,
    credential: ClaudeCredential,
    default_model: str | None = None,
    tier_model: str | None = None,
) -> Mapping[str, JsonValue]:
    """Return a new settings mapping wired to route Claude Code through the proxy.

    A StaticToken (a long-lived virtual key, or a local master key) is written into
    env.ANTHROPIC_AUTH_TOKEN; an ApiKeyHelper (the short-lived `lite login` credential, which
    Claude Code re-reads through `lite auth print-token` on every request) becomes the top-level
    apiKeyHelper. Whichever is written, the other credential slots, env.ANTHROPIC_API_KEY, a stale
    env.ANTHROPIC_AUTH_TOKEN and a stale apiKeyHelper, are removed, since Claude Code given two
    credentials at once may send the wrong one. ENABLE_TOOL_SEARCH defaults to true because Claude
    Code turns tool search off when ANTHROPIC_BASE_URL is not a first-party Anthropic host, and
    CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY defaults to 1 so the /model picker is filled from
    the proxy's /v1/models; an existing value of either is kept.

    `default_model` becomes the top-level `model`, the row Claude Code starts on and shows as
    "from settings.json"; the /model picker still lists every discovered model. `tier_model`
    is `lite autoroute up`'s knob: it sets every ANTHROPIC_DEFAULT_*_MODEL so Claude Code's own
    /model aliases, sub-agents and background helpers all request that one group instead of
    Claude Code's built-in ids. Apart from those tier keys, the keys this touches are exactly
    OWNED_ENV_KEYS and OWNED_TOP_LEVEL_KEYS; every other key is preserved untouched.
    """
    raw_env: Final = settings.get(ENV_KEY, {})
    current_env: Final = raw_env if isinstance(raw_env, dict) else {}
    env: Final = dict(  # mutable-ok: JSON document handed to json.dump, which rejects a read-only mapping
        chain(
            (
                (ENABLE_TOOL_SEARCH_KEY, ENABLE_TOOL_SEARCH_VALUE),
                (ENABLE_GATEWAY_MODEL_DISCOVERY_KEY, ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE),
            ),
            ((key, value) for key, value in current_env.items() if key not in _CREDENTIAL_ENV_KEYS),
            ((ANTHROPIC_BASE_URL_KEY, base_url.rstrip("/")),),
            ((ANTHROPIC_AUTH_TOKEN_KEY, credential.token),) if isinstance(credential, StaticToken) else (),
            ((key, tier_model) for key in ANTHROPIC_DEFAULT_MODEL_ENV_KEYS if tier_model is not None),
        )
    )
    return dict(  # mutable-ok: JSON document handed to json.dump, which rejects a read-only mapping
        chain(
            ((key, value) for key, value in settings.items() if key not in (API_KEY_HELPER_KEY, ENV_KEY)),
            ((ENV_KEY, env),),
            ((API_KEY_HELPER_KEY, credential.command),) if isinstance(credential, ApiKeyHelper) else (),
            ((MODEL_KEY, default_model),) if default_model is not None else (),
        )
    )


def resolve_api_key_helper(base_url: str, platform: str = sys.platform) -> str:
    """Build the shell command Claude Code should run for its apiKeyHelper.

    Claude Code hands the string to the system shell, `sh` on POSIX and cmd.exe
    on Windows, so every token is quoted for the shell that will read it.

    Resolves `lite` to an absolute path so the helper works regardless of the
    PATH visible to whatever subprocess Claude Code spawns it from. Passing
    --base-url explicitly (rather than relying on the bare invocation Claude
    Code would otherwise use) makes `print-token` enforce that the cached
    token was actually issued for this proxy -- without it, a token minted
    for a different, previously-logged-into proxy would be handed to
    whichever server the settings currently point at.

    --base-url belongs to the top-level `lite` group, so it has to precede the
    subcommand; click rejects it outright after `print-token`.
    """
    lite_path: Final = shutil.which("lite")
    if lite_path is None:
        raise ClaudeSettingsError(
            "Could not find `lite` on your PATH. Claude Code's apiKeyHelper needs an absolute path to it."
        )
    quote: Final = quote_for_cmd if platform.startswith("win") else shlex.quote
    return " ".join(quote(token) for token in (lite_path, "--base-url", base_url, "auth", "print-token"))


def _owned(container: Mapping[str, JsonValue], key: str) -> OwnedValue:
    return OwnedValue(present=key in container, value=container.get(key))


def _fingerprint(owned: OwnedValue) -> str:
    return hashlib.sha256(json.dumps(owned.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def _snapshot(container: Mapping[str, JsonValue], keys: Sequence[str]) -> Mapping[str, OwnedValue]:
    return MappingProxyType({key: _owned(container, key) for key in keys})


def _fingerprints(container: Mapping[str, JsonValue], keys: Sequence[str]) -> Mapping[str, str]:
    return MappingProxyType({key: _fingerprint(_owned(container, key)) for key in keys})


def read_configure_receipt(state_path: Path) -> ConfigureReceipt | None:
    if not state_path.exists():
        return None
    try:
        return ConfigureReceipt.model_validate_json(state_path.read_bytes())
    except (OSError, ValidationError) as e:
        raise ClaudeSettingsError(
            f"{state_path} is not a readable `lite configure claude` receipt ({e}). "
            "Remove it and edit Claude Code's settings by hand if they still point at the proxy."
        ) from e


def configure_claude_settings(
    base_url: str,
    credential: ClaudeCredential,
    model: ModelChoice,
    settings_path: Path,
    state_path: Path,
    owners: Sequence[SettingsFileOwner],
    commit: Callable[[str, str], None] = commit_staged_json,
) -> None:
    """Persistently route Claude Code through base_url, recording how to undo it.

    Both files are staged before either is committed, so a full disk or a read-only directory
    fails before anything changes. The two commits are still two renames, so if the settings
    rename fails after the receipt landed, the earlier receipt is put back (or the new one
    removed on a first configure): the receipt on disk never describes settings that were not
    written. A repeat configure keeps the receipt's original `previous_*`
    snapshot and only refreshes what was written, so unconfigure still returns to the
    pre-configure state. `model` says what happens to the top-level model: StartOn pins it,
    UnpinModel lets go of a pin an earlier configure made (back to whatever the user had, never
    keeping it silently), and KeepModel leaves it alone, which is what a re-login wants.
    """
    _refuse_while_owned(settings_path, owners)
    current: Final = load_json_or_empty(settings_path)
    earlier: Final = read_configure_receipt(state_path)
    existing: Final = (
        _restored(current, (MODEL_KEY,), earlier.previous_top_level, earlier.written_top_level)[0]
        if isinstance(model, UnpinModel) and earlier is not None
        else current
    )
    previous_env: Final = _env_object(existing, settings_path)
    merged: Final = merge_claude_settings(
        existing, base_url, credential, model.model if isinstance(model, StartOn) else None
    )
    receipt: Final = ConfigureReceipt(
        file_existed=earlier.file_existed if earlier is not None else settings_path.exists(),
        env_present=earlier.env_present if earlier is not None else ENV_KEY in existing,
        env_was_object=earlier.env_was_object if earlier is not None else isinstance(existing.get(ENV_KEY), dict),
        previous_env=earlier.previous_env if earlier is not None else _snapshot(previous_env, OWNED_ENV_KEYS),
        previous_top_level=earlier.previous_top_level
        if earlier is not None
        else _snapshot(existing, OWNED_TOP_LEVEL_KEYS),
        written_env=_fingerprints(_env_object(merged, settings_path), OWNED_ENV_KEYS),
        written_top_level=_fingerprints(merged, OWNED_TOP_LEVEL_KEYS),
    )
    target: Final = _write_target(settings_path)
    try:
        ensure_private_dir(state_path.parent)
    except OSError as e:
        raise ClaudeSettingsError(f"Could not write {state_path}: {e}") from e
    staged_receipt: Final = _stage(state_path, receipt.model_dump(mode="json"))
    try:
        staged_settings: Final = _stage(target, merged)
    except ClaudeSettingsError:
        discard_staged_json(staged_receipt)
        raise
    commit(staged_receipt, str(state_path))
    try:
        commit(staged_settings, str(target))
    except OSError as e:
        _restore_receipt(state_path, earlier)
        raise ClaudeSettingsError(f"Could not write {target}: {e}") from e


def _restore_receipt(state_path: Path, earlier: ConfigureReceipt | None) -> None:
    if earlier is None:
        state_path.unlink(missing_ok=True)
        return
    commit_staged_json(stage_private_json(str(state_path), earlier.model_dump(mode="json")), str(state_path))


def _restored(
    current: Mapping[str, JsonValue],
    keys: Sequence[str],
    previous: Mapping[str, OwnedValue],
    written: Mapping[str, str],
) -> tuple[Mapping[str, JsonValue], tuple[str, ...], tuple[str, ...]]:
    """Put back every owned key that still holds what configure wrote; leave the rest alone."""
    ours: Final = frozenset(key for key in keys if _fingerprint(_owned(current, key)) == written[key])
    restored: Final = dict(  # mutable-ok: JSON document handed to json.dump, which rejects a read-only mapping
        chain(
            ((key, value) for key, value in current.items() if key not in ours),
            ((key, previous[key].value) for key in keys if key in ours and previous[key].present),
        )
    )
    return restored, tuple(key for key in keys if key in ours), tuple(key for key in keys if key not in ours)


def _without(previous: Mapping[str, OwnedValue], keys: frozenset[str]) -> Mapping[str, OwnedValue]:
    """The previous snapshot with the named keys recorded as absent, so restoring them removes them.

    A credential is only ever put back next to the endpoint it was issued for: once the user has
    pointed ANTHROPIC_BASE_URL somewhere else, the old ANTHROPIC_API_KEY or apiKeyHelper would be
    sent to that new server on the next request, so those slots stay removed instead.
    """
    return MappingProxyType(
        {key: OwnedValue(present=False) if key in keys else value for key, value in previous.items()}
    )


def _restored_env(env: Mapping[str, JsonValue], receipt: ConfigureReceipt) -> OwnedValue:
    """An env object that existed before stays an object; one that configure created goes back to
    absent or null once it is empty again, and stays when the user has since put keys in it."""
    if receipt.env_was_object or env:
        return OwnedValue(present=True, value=dict(env))  # mutable-ok: JSON document handed to json.dump
    return OwnedValue(present=receipt.env_present, value=None)


def unconfigure_claude_settings(
    settings_path: Path, state_path: Path, owners: Sequence[SettingsFileOwner]
) -> UnconfigureOutcome:
    """Undo `lite configure claude`, restoring only the keys the user has not changed since.

    `withheld` names the credentials that were ours to restore and were deliberately left removed
    because the base URL is no longer ours; a credential slot the user changed since is `kept`, never
    both.
    """
    _refuse_while_owned(settings_path, owners)
    receipt: Final = read_configure_receipt(state_path)
    if receipt is None:
        raise ClaudeSettingsError(
            f"Claude Code is not configured by `lite configure claude` (no receipt at {state_path}); nothing to undo."
        )
    current: Final = load_json_or_empty(settings_path)
    current_env: Final = _env_object(current, settings_path)
    url_is_ours: Final = (
        _fingerprint(_owned(current_env, ANTHROPIC_BASE_URL_KEY)) == receipt.written_env[ANTHROPIC_BASE_URL_KEY]
    )
    previous_env: Final = receipt.previous_env if url_is_ours else _without(receipt.previous_env, _CREDENTIAL_ENV_KEYS)
    previous_top: Final = (
        receipt.previous_top_level
        if url_is_ours
        else _without(receipt.previous_top_level, frozenset((API_KEY_HELPER_KEY,)))
    )
    env, restored_env, kept_env = _restored(current_env, OWNED_ENV_KEYS, previous_env, receipt.written_env)
    top_level, restored_top, kept_top = _restored(
        current, OWNED_TOP_LEVEL_KEYS, previous_top, receipt.written_top_level
    )
    withheld: Final = (
        ()
        if url_is_ours
        else (
            *(
                f"{ENV_KEY}.{key}"
                for key in restored_env
                if key in _CREDENTIAL_ENV_KEYS and receipt.previous_env[key].present
            ),
            *(
                (API_KEY_HELPER_KEY,)
                if API_KEY_HELPER_KEY in restored_top and receipt.previous_top_level[API_KEY_HELPER_KEY].present
                else ()
            ),
        )
    )
    env_restored: Final = _restored_env(env, receipt)
    settings: Final = dict(  # mutable-ok: JSON document handed to json.dump, which rejects a read-only mapping
        chain(
            ((key, value) for key, value in top_level.items() if key != ENV_KEY),
            () if env_restored.present is False else ((ENV_KEY, env_restored.value),),
        )
    )
    if settings or receipt.file_existed:
        _write_settings(settings_path, settings)
    else:
        _write_target(settings_path).unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    return UnconfigureOutcome(
        restored=(
            *(f"{ENV_KEY}.{key}" for key in restored_env if f"{ENV_KEY}.{key}" not in withheld),
            *(key for key in restored_top if key not in withheld),
        ),
        kept=(*(f"{ENV_KEY}.{key}" for key in kept_env), *kept_top),
        withheld=withheld,
    )


__all__ = (
    "ANTHROPIC_API_KEY_KEY",
    "ANTHROPIC_AUTH_TOKEN_KEY",
    "ANTHROPIC_BASE_URL_KEY",
    "ANTHROPIC_DEFAULT_MODEL_ENV_KEYS",
    "API_KEY_HELPER_KEY",
    "AUTOROUTE_BACKUP_PATH",
    "BACKUP_PATH",
    "CLAUDE_SETTINGS_PATH",
    "CONFIGURE_STATE_PATH",
    "ENABLE_GATEWAY_MODEL_DISCOVERY_KEY",
    "ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE",
    "ENABLE_TOOL_SEARCH_KEY",
    "ENABLE_TOOL_SEARCH_VALUE",
    "ENV_KEY",
    "MODEL_KEY",
    "OWNED_ENV_KEYS",
    "OWNED_TOP_LEVEL_KEYS",
    "SETTINGS_FILE_OWNERS",
    "ApiKeyHelper",
    "ClaudeCredential",
    "ClaudeSettingsError",
    "ConfigureReceipt",
    "KeepModel",
    "ModelChoice",
    "OwnedValue",
    "SettingsFileOwner",
    "StartOn",
    "StaticToken",
    "UnconfigureOutcome",
    "UnpinModel",
    "configure_claude_settings",
    "load_json_or_empty",
    "merge_claude_settings",
    "read_configure_receipt",
    "resolve_api_key_helper",
    "unconfigure_claude_settings",
)
