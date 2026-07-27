# Extending FaceGuard: Providers and Triggers

FaceGuard has three pluggable extension points. Each is one small in-tree
registry (or, for triggers, an explicit map) — no plugin discovery, no
dynamic imports of arbitrary paths:

| Extension point | Protocol / base class | Registered by | Selected by |
|---|---|---|---|
| Local (embedding) provider | `EmbeddingProvider` (`src/providers/interfaces.py`) | `@register_local(name)` | `LOCAL_PROVIDER` |
| Cloud match provider | `CloudMatchProvider` (`src/providers/interfaces.py`) | `@register_cloud(name)` | `CLOUD_PROVIDER` |
| On-match trigger | `Trigger` (`src/triggers/base.py`) | entry in `create_trigger()`'s if-chain | `TRIGGER_PROVIDER` |

## Two hard facts before you start

1. **`embedding_dim` must be 512.** The `faces.embedding_local` column is a
   pgvector `Vector(512)` (`src/database/models.py`). Startup validation
   (`src/utils/startup_validation.py::validate_embedding_dim`) raises
   `RuntimeError` if the resolved local provider's `embedding_dim != 512`:

   ```
   Local provider '<name>' has embedding_dim=<n>; the faces.embedding_local
   column is Vector(512). A different dimension needs its own migration.
   ```

   A different dimension is possible, but it's a schema change — a new
   `alembic revision` altering the column type (and reindexing) — not a
   config flag.

2. **Never `issubclass()` against these Protocols — `isinstance()` on
   instances only.** `EmbeddingProvider` and `CloudMatchProvider` are
   `@runtime_checkable` Protocols with non-method members (`name`,
   `embedding_dim`). `runtime_checkable` only supports `isinstance()` checks
   when a Protocol has non-method members; `issubclass(SomeClass,
   EmbeddingProvider)` raises `TypeError: Protocols with non-method members
   don't support issubclass()`. The contract tests below already do this
   correctly (`assert isinstance(provider, EmbeddingProvider)`) — copy that
   pattern.

---

## Adding a local (embedding) provider

A local provider turns image bytes into a 512-d face embedding, in-process
(no network round trip). This is the `EmbeddingProvider` Protocol, copied
verbatim from `src/providers/interfaces.py`:

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    embedding_dim: int

    async def extract_embedding(self, image_bytes: bytes) -> np.ndarray: ...

    async def detect_multiple_faces(self, image_bytes: bytes) -> list: ...
```

### Skeleton implementation

```python
# src/providers/my_provider.py
from __future__ import annotations

import numpy as np

from src.providers.registry import register_local


@register_local("my_provider")
class MyProvider:
    name = "my_provider"
    embedding_dim = 512  # must match faces.embedding_local's Vector(512)

    async def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        # Return a (512,) L2-normalized float array.
        raise NotImplementedError

    async def detect_multiple_faces(self, image_bytes: bytes) -> list:
        # Return a list of per-face detections; see
        # src/providers/insightface_provider.py for a full reference shape.
        raise NotImplementedError
```

The `@register_local("my_provider")` decorator just stamps `cls.name` and
stores the class in an in-memory dict — it does **not** make the module get
imported. Registration only runs when the module is imported, so add it to
`_ensure_builtins_loaded()` in `src/providers/registry.py`:

```python
def _ensure_builtins_loaded() -> None:
    # Import side effects perform registration; guarded to avoid cycles.
    from src.providers import (  # noqa: F401
        aws_rekognition,
        insightface_provider,
        my_provider,          # <-- add your module
        silent_face_liveness,
    )
```

Without that import, `LOCAL_PROVIDER=my_provider` fails at startup with
`ConfigurationError: Unknown local provider 'my_provider'`.

> **Note:** `get_local_provider()` in `src/providers/factory.py` currently
> instantiates whichever class is resolved with InsightFace's constructor
> keywords (`model_name`, `det_size`, `ctx_id`). Give your `__init__` the
> same signature (even if it ignores them) until a second local provider
> forces that factory to become provider-agnostic.

### Contract test

Subclass `EmbeddingProviderContract` in your own test file — the shared
`sample_image_bytes` fixture comes from `tests/contracts/conftest.py`
automatically, no need to redefine it:

```python
# tests/contracts/test_my_provider_contract.py
import pytest

from tests.contracts.test_provider_contracts import EmbeddingProviderContract


class TestMyProviderContract(EmbeddingProviderContract):
    @pytest.fixture
    def provider(self):
        from src.providers.my_provider import MyProvider

        return MyProvider()
```

This inherits `test_satisfies_protocol` (`isinstance` check),
`test_declares_512_dim`, and
`test_extract_embedding_returns_normalized_512` unchanged.

### Env var

```
LOCAL_PROVIDER=my_provider   # default: insightface
```

---

## Adding a cloud match provider

A cloud match provider enrolls/searches/compares faces via an external
service's own face collection (no local embedding exposed). This is the
`CloudMatchProvider` Protocol, copied verbatim from
`src/providers/interfaces.py`:

```python
@runtime_checkable
class CloudMatchProvider(Protocol):
    name: str

    async def enroll_face(self, image_bytes: bytes, metadata: FaceMetadata) -> EnrollmentResult: ...

    async def recognize_face(
        self, image_bytes: bytes, max_results: int, confidence_threshold: float
    ) -> list[FaceMatch]: ...

    async def compare_faces(self, source_bytes: bytes, target_bytes: bytes) -> float | None: ...

    async def delete_face(self, face_id: str, collection_id: str | None = None) -> bool: ...
```

`FaceMatch`, `EnrollmentResult`, and `FaceMetadata` are plain dataclasses in
`src/providers/base.py`.

### Skeleton implementation

```python
# src/providers/my_cloud.py
from __future__ import annotations

from src.providers.base import EnrollmentResult, FaceMatch, FaceMetadata
from src.providers.registry import register_cloud


@register_cloud("my_cloud")
class MyCloudProvider:
    name = "my_cloud"

    async def enroll_face(self, image_bytes: bytes, metadata: FaceMetadata) -> EnrollmentResult:
        raise NotImplementedError

    async def recognize_face(
        self, image_bytes: bytes, max_results: int, confidence_threshold: float
    ) -> list[FaceMatch]:
        raise NotImplementedError

    async def compare_faces(self, source_bytes: bytes, target_bytes: bytes) -> float | None:
        raise NotImplementedError

    async def delete_face(self, face_id: str, collection_id: str | None = None) -> bool:
        raise NotImplementedError
```

Same wiring rule as local providers — add the import to
`_ensure_builtins_loaded()` in `src/providers/registry.py` so the decorator
runs before `CLOUD_PROVIDER=my_cloud` gets resolved:

```python
def _ensure_builtins_loaded() -> None:
    from src.providers import (  # noqa: F401
        aws_rekognition,
        insightface_provider,
        my_cloud,              # <-- add your module
        silent_face_liveness,
    )
```

`get_cloud_provider()` in `src/providers/factory.py` instantiates with no
arguments (`provider_class()`), so keep `__init__` argument-free (or
all-default).

### Contract test

```python
# tests/contracts/test_my_cloud_contract.py
import pytest

from tests.contracts.test_provider_contracts import CloudMatchProviderContract


class TestMyCloudContract(CloudMatchProviderContract):
    @pytest.fixture
    def provider(self):
        from src.providers.my_cloud import MyCloudProvider

        return MyCloudProvider()
```

This inherits `test_satisfies_protocol` and
`test_compare_faces_returns_unit_interval` unchanged — again using the
shared `sample_image_bytes` fixture from `tests/contracts/conftest.py`.

### Env var

```
CLOUD_PROVIDER=my_cloud   # default: aws_rekognition
```

---

## Adding a trigger

A trigger fires a side effect (log line, webhook POST, GPIO pulse, ...) when
a recognition crosses `TRIGGER_CONFIDENCE_THRESHOLD`. This is the `Trigger`
ABC and its companion dataclasses, copied verbatim from
`src/triggers/base.py`:

```python
@dataclass
class MatchEvent:
    user_name: str
    confidence: float
    processor: str
    user_email: str | None = None
    camera_id: int | None = None
    liveness_passed: bool | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(frozen=True)
class TriggerResult:
    success: bool
    action: str
    detail: str | None = None


class Trigger(ABC):
    name: str = "base"

    @abstractmethod
    async def fire(self, event: MatchEvent) -> TriggerResult: ...
```

Triggers have no decorator-based registry — `create_trigger()` in
`src/triggers/providers.py` is a plain if-chain. Implement `fire()` and add
a branch:

### Skeleton implementation

```python
# src/triggers/providers.py (or your own module, imported from here)
class MyTrigger(Trigger):
    name = "my_trigger"

    async def fire(self, event: MatchEvent) -> TriggerResult:
        # e.g. push to a message queue, ring a bell, whatever "my_trigger" means.
        return TriggerResult(success=True, action="fired")
```

```python
def create_trigger(name: str | None = None) -> Trigger:
    resolved = _LEGACY_NAMES.get(
        name or settings.trigger_provider, name or settings.trigger_provider
    )
    if resolved == "log":
        return LogTrigger()
    if resolved == "webhook":
        if not settings.trigger_webhook_url:
            raise ValueError("TRIGGER_WEBHOOK_URL must be set for the webhook trigger")
        return WebhookTrigger(url=settings.trigger_webhook_url)
    if resolved == "gpio":
        return GpioTrigger(pin=settings.trigger_gpio_pin)
    if resolved == "my_trigger":
        return MyTrigger()
    raise ValueError(f"Unknown trigger provider: {resolved!r}")
```

### Test

There's no shared `TriggerContract` base class — write a direct unit test,
same shape as the existing ones in `tests/unit/test_triggers.py`:

```python
from src.triggers.base import MatchEvent, TriggerResult
from src.triggers.providers import MyTrigger


def _event(confidence=0.95):
    return MatchEvent(user_name="alice", confidence=confidence, processor="antelopev2")


class TestMyTrigger:
    async def test_fires(self):
        result = await MyTrigger().fire(_event())
        assert result == TriggerResult(success=True, action="fired")
```

### Env var

```
TRIGGER_PROVIDER=my_trigger   # default: log
```

---

## See also

- [`docs/configuration.md`](configuration.md) — the "Providers (advanced)" table and the full env var reference.
- `tests/contracts/` — the contract base classes (`test_provider_contracts.py`) and the shared `sample_image_bytes` fixture (`conftest.py`).
- `tests/unit/test_triggers.py` — existing trigger unit tests to pattern-match.
