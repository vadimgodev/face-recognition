"""In-tree provider registry. A provider = one class + one register decorator."""

from __future__ import annotations

from src.exceptions import ConfigurationError

_LOCAL: dict[str, type] = {}
_CLOUD: dict[str, type] = {}
_LIVENESS: dict[str, type] = {}


def _register(store: dict[str, type], name: str):
    def deco(cls: type) -> type:
        cls.name = name  # type: ignore[attr-defined]  # decorator injects provider name
        store[name] = cls
        return cls

    return deco


def register_local(name: str):
    return _register(_LOCAL, name)


def register_cloud(name: str):
    return _register(_CLOUD, name)


def register_liveness(name: str):
    return _register(_LIVENESS, name)


def _resolve(store: dict[str, type], name: str, role: str) -> type:
    _ensure_builtins_loaded()
    if name not in store:
        raise ConfigurationError(f"Unknown {role} provider {name!r}; available: {sorted(store)}")
    return store[name]


def _ensure_builtins_loaded() -> None:
    # Import side effects perform registration; guarded to avoid cycles.
    from src.providers import (  # noqa: F401
        aws_rekognition,
        insightface_provider,
        silent_face_liveness,
    )


def resolve_local(name: str) -> type:
    return _resolve(_LOCAL, name, "local")


def resolve_cloud(name: str) -> type:
    return _resolve(_CLOUD, name, "cloud")


def resolve_liveness(name: str) -> type:
    return _resolve(_LIVENESS, name, "liveness")


def available_local() -> list[str]:
    _ensure_builtins_loaded()
    return sorted(_LOCAL)


def available_cloud() -> list[str]:
    _ensure_builtins_loaded()
    return sorted(_CLOUD)
