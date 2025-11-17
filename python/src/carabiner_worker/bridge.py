from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable


@runtime_checkable
class _BridgeProtocol(Protocol):
    def run_instance(self, database_url: str, payload: bytes) -> int: ...

    def wait_for_instance(
        self, database_url: str, poll_interval_secs: float
    ) -> Optional[bytes]: ...


if TYPE_CHECKING:  # pragma: no cover
    from . import _bridge as _native_bridge  # type: ignore[attr-defined]
else:
    import importlib

    try:
        _native_bridge = importlib.import_module("carabiner_worker._bridge")
    except ImportError:  # pragma: no cover
        _native_bridge = None

_bridge: Optional[_BridgeProtocol] = _native_bridge


def _require_bridge() -> _BridgeProtocol:
    if _bridge is None:
        raise RuntimeError(
            "carabiner_worker._bridge is unavailable - rebuild the project with maturin to enable the Rust client bridge"
        )
    return _bridge


def run_instance(database_url: str, payload: bytes) -> int:
    """Invoke the Rust client bridge to enqueue a workflow instance."""
    bridge = _require_bridge()
    return bridge.run_instance(database_url, payload)


def wait_for_instance(database_url: str, poll_interval_secs: float = 1.0) -> Optional[bytes]:
    """Poll the database via Rust bridge until an instance result is available."""
    bridge = _require_bridge()
    return bridge.wait_for_instance(database_url, poll_interval_secs)
