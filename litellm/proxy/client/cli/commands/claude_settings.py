"""Claude Code's ~/.claude/settings.json: what `lite` writes there and how it undoes it.

`lite up` and `lite autoroute up` patch this file temporarily and restore a backup on exit;
`lite login --config-claude` and `lite configure claude` patch it persistently through the
receipt in `agent_config`. All of them share one merge and one apiKeyHelper command, and `up`
already imports from `auth`, so the shared parts live here rather than in any one command module.
"""

import os
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import JsonValue

from litellm.litellm_core_utils.private_json import commit_staged_json

from .agent_config import (
    ROOT_SECTION,
    AgentConfigError,
    ConfigDocument,
    JsonDocument,
    RestoreGroup,
    SettingsFileOwner,
    UnconfigureOutcome,
    configure_document,
    read_bytes_or_empty,
    unconfigure_document,
)
from .cmd_quoting import quote_for_cmd

ClaudeSettingsError: TypeAlias = AgentConfigError

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
OWNED_SECTIONS: Final[Mapping[str, Sequence[str]]] = MappingProxyType(
    {ROOT_SECTION: OWNED_TOP_LEVEL_KEYS, ENV_KEY: OWNED_ENV_KEYS}
)
_CREDENTIAL_ENV_KEYS: Final = frozenset((ANTHROPIC_API_KEY_KEY, ANTHROPIC_AUTH_TOKEN_KEY))
CREDENTIAL_RESTORE_GROUP: Final = RestoreGroup(
    anchor=(ENV_KEY, ANTHROPIC_BASE_URL_KEY),
    dependents=(
        (ENV_KEY, ANTHROPIC_API_KEY_KEY),
        (ENV_KEY, ANTHROPIC_AUTH_TOKEN_KEY),
        (ROOT_SECTION, API_KEY_HELPER_KEY),
    ),
)
CLAUDE_CONFIG_DIR_ENV: Final = "CLAUDE_CONFIG_DIR"


def claude_config_dir() -> Path:
    """Where Claude Code keeps its config: ~/.claude unless CLAUDE_CONFIG_DIR relocates it."""
    override: Final = os.environ.get(CLAUDE_CONFIG_DIR_ENV)
    return Path(override).expanduser() if override else Path.home() / ".claude"


CLAUDE_SETTINGS_PATH: Final = claude_config_dir() / "settings.json"
BACKUP_PATH: Final = Path.home() / ".litellm" / "claude_settings_backup.json"
AUTOROUTE_BACKUP_PATH: Final = Path.home() / ".litellm" / "autorouter" / "claude_settings_backup.json"
CONFIGURE_STATE_PATH: Final = Path.home() / ".litellm" / "claude_configure_state.json"

SETTINGS_FILE_OWNERS: Final = (
    SettingsFileOwner(BACKUP_PATH, "lite up", "lite down"),
    SettingsFileOwner(AUTOROUTE_BACKUP_PATH, "lite autoroute up", "lite autoroute down"),
)


@dataclass(frozen=True, slots=True)
class StaticToken:
    """A long-lived virtual key, written into the agent's config as a bearer token."""

    token: str


@dataclass(frozen=True, slots=True)
class ApiKeyHelper:
    """A `lite auth print-token` invocation the agent runs per request, so a login renews in place."""

    command: str


ClaudeCredential: TypeAlias = StaticToken | ApiKeyHelper


@dataclass(frozen=True, slots=True)
class KeepModel:
    """Leave the pinned model as it is, the user's or an earlier configure's (a re-login)."""


@dataclass(frozen=True, slots=True)
class UnpinModel:
    """Let go of a model an earlier configure pinned; one the user set themselves stays."""


@dataclass(frozen=True, slots=True)
class StartOn:
    """Pin the model the agent starts on."""

    model: str


ModelChoice: TypeAlias = KeepModel | UnpinModel | StartOn


def load_json_or_empty(path: Path) -> dict[str, JsonValue]:
    return dict(JsonDocument.parse(read_bytes_or_empty(path), path).root())  # mutable-ok: callers hand it to json.dump


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
    is `lite autoroute up`'s knob (passed together with `default_model`): it sets every
    ANTHROPIC_DEFAULT_*_MODEL so Claude Code's own /model aliases, sub-agents and background
    helpers all request that one group instead of Claude Code's built-in ids. Apart from those
    tier keys, the keys this touches are exactly OWNED_ENV_KEYS and OWNED_TOP_LEVEL_KEYS; every
    other key is preserved untouched.
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


def _apply_claude_merge(
    document: ConfigDocument, base_url: str, credential: ClaudeCredential, model: str | None
) -> None:
    root: Final = document.section(ROOT_SECTION)
    merged: Final = merge_claude_settings(
        root if root is not None else MappingProxyType({}), base_url, credential, model
    )
    for key in OWNED_TOP_LEVEL_KEYS:
        if key in merged:
            document.set_value(ROOT_SECTION, key, merged[key])
        else:
            document.delete(ROOT_SECTION, key)
    env: Final = merged[ENV_KEY]
    if isinstance(env, dict):
        for key in OWNED_ENV_KEYS:
            if key in env:
                document.set_value(ENV_KEY, key, env[key])
            else:
                document.delete(ENV_KEY, key)


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
    quote: Final = quote_for_cmd if platform.startswith("win") else shlex.quote
    return " ".join(quote(token) for token in print_token_command(base_url))


def print_token_command(base_url: str) -> tuple[str, ...]:
    """The argv Claude Code's apiKeyHelper and Codex's provider auth command both run."""
    lite_path: Final = shutil.which("lite")
    if lite_path is None:
        raise AgentConfigError(
            "Could not find `lite` on your PATH. The agent's credential command needs an absolute path to it."
        )
    return (lite_path, "--base-url", base_url, "auth", "print-token")


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

    `model` says what happens to the top-level model: StartOn pins it, UnpinModel lets go of a
    pin an earlier configure made (back to whatever the user had, never keeping it silently),
    and KeepModel leaves it alone, which is what a re-login wants.
    """
    configure_document(
        settings_path,
        state_path,
        owners,
        JsonDocument.parse,
        OWNED_SECTIONS,
        lambda document: _apply_claude_merge(
            document, base_url, credential, model.model if isinstance(model, StartOn) else None
        ),
        release=((ROOT_SECTION, MODEL_KEY),) if isinstance(model, UnpinModel) else (),
        commit=commit,
    )


def unconfigure_claude_settings(
    settings_path: Path, state_path: Path, owners: Sequence[SettingsFileOwner]
) -> UnconfigureOutcome:
    """Undo `lite configure claude` (and `lite login --config-claude`), restoring only unchanged keys."""
    return unconfigure_document(
        settings_path, state_path, owners, JsonDocument.parse, "Claude Code", groups=(CREDENTIAL_RESTORE_GROUP,)
    )


__all__ = (
    "ANTHROPIC_API_KEY_KEY",
    "ANTHROPIC_AUTH_TOKEN_KEY",
    "ANTHROPIC_BASE_URL_KEY",
    "ANTHROPIC_DEFAULT_MODEL_ENV_KEYS",
    "API_KEY_HELPER_KEY",
    "AUTOROUTE_BACKUP_PATH",
    "BACKUP_PATH",
    "CLAUDE_CONFIG_DIR_ENV",
    "CLAUDE_SETTINGS_PATH",
    "CONFIGURE_STATE_PATH",
    "ENABLE_GATEWAY_MODEL_DISCOVERY_KEY",
    "ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE",
    "ENABLE_TOOL_SEARCH_KEY",
    "ENABLE_TOOL_SEARCH_VALUE",
    "ENV_KEY",
    "MODEL_KEY",
    "OWNED_ENV_KEYS",
    "OWNED_SECTIONS",
    "OWNED_TOP_LEVEL_KEYS",
    "SETTINGS_FILE_OWNERS",
    "ApiKeyHelper",
    "ClaudeCredential",
    "ClaudeSettingsError",
    "KeepModel",
    "ModelChoice",
    "SettingsFileOwner",
    "StartOn",
    "StaticToken",
    "UnconfigureOutcome",
    "UnpinModel",
    "claude_config_dir",
    "configure_claude_settings",
    "load_json_or_empty",
    "merge_claude_settings",
    "print_token_command",
    "resolve_api_key_helper",
    "unconfigure_claude_settings",
)
