"""Console script shims for the bundled Rust binaries."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _binary_path(name: str) -> Path:
    binary = Path(__file__).resolve().parent / "bin" / name
    if not binary.exists():  # pragma: no cover - packaging guard
        raise FileNotFoundError(f"missing rappel binary: {binary}")
    return binary


def _exec_binary(name: str) -> None:
    binary = _binary_path(name)
    argv = [binary.name, *sys.argv[1:]]
    os.execv(str(binary), argv)


def rappel_server() -> None:
    _exec_binary("rappel-server")


def start_workers() -> None:
    _exec_binary("start_workers")


def boot_rappel_singleton() -> None:
    _exec_binary("boot-rappel-singleton")
