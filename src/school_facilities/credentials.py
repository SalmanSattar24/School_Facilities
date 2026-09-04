from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


class CredentialError(RuntimeError):
    pass


def load_api_key(
    *,
    environment_variable: str,
    secrets_file: Path,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    active_environment = os.environ if environment is None else environment
    environment_value = active_environment.get(environment_variable, "").strip()
    if environment_value:
        return environment_value, "environment"

    try:
        lines = secrets_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    except OSError as error:
        raise CredentialError(f"local secrets file could not be read: {error}") from error
    prefix = f"{environment_variable}="
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or not stripped.startswith(prefix):
            continue
        stored_value = stripped[len(prefix):].strip()
        if (
            len(stored_value) >= 2
            and stored_value[0] == stored_value[-1]
            and stored_value[0] in {'"', "'"}
        ):
            stored_value = stored_value[1:-1]
        if stored_value.strip():
            return stored_value.strip(), "local_secrets_file"
    raise CredentialError(
        f"no Gemini key found in {environment_variable} or {secrets_file}"
    )


def save_api_key(
    api_key: str,
    *,
    environment_variable: str,
    secrets_file: Path,
) -> None:
    value = api_key.strip()
    if not value:
        raise CredentialError("Gemini API key must not be blank")
    try:
        secrets_file.write_text(f"{environment_variable}={value}\n", encoding="utf-8")
        try:
            secrets_file.chmod(0o600)
        except OSError:
            pass
    except OSError as error:
        raise CredentialError(f"local secrets file could not be written: {error}") from error


def upsert_api_key(
    api_key: str,
    *,
    environment_variable: str,
    secrets_file: Path,
    provider_name: str,
) -> None:
    """Add or replace one key without deleting other local credentials."""
    value = api_key.strip()
    if not value:
        raise CredentialError(f"{provider_name} API key must not be blank")
    try:
        existing = secrets_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        existing = []
    except OSError as error:
        raise CredentialError(f"local secrets file could not be read: {error}") from error
    prefix = f"{environment_variable}="
    output: list[str] = []
    replaced = False
    for line in existing:
        if line.strip().startswith(prefix):
            if not replaced:
                output.append(f"{environment_variable}={value}")
                replaced = True
            continue
        output.append(line)
    if not replaced:
        if output and output[-1].strip():
            output.append("")
        output.append(f"{environment_variable}={value}")
    try:
        secrets_file.write_text("\n".join(output) + "\n", encoding="utf-8")
        try:
            secrets_file.chmod(0o600)
        except OSError:
            pass
    except OSError as error:
        raise CredentialError(f"local secrets file could not be written: {error}") from error
