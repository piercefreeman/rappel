#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Build a npm bundle that includes the JS package and native binaries."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EntryPoint:
    built_name: str
    packaged_name: str


ENTRYPOINTS = (
    EntryPoint("rappel-bridge", "rappel-bridge"),
    EntryPoint("boot-rappel-singleton", "boot-rappel-singleton"),
    EntryPoint("start-workers", "start-workers"),
)


def run(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_binaries(repo_root: Path, stage_dir: Path) -> list[Path]:
    target_dir = repo_root / "target" / "release"
    stage_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    suffix = ".exe" if sys.platform == "win32" else ""
    for entry in ENTRYPOINTS:
        src = target_dir / f"{entry.built_name}{suffix}"
        if not src.exists():
            raise FileNotFoundError(f"Missing compiled binary: {src}")
        dest = stage_dir / f"{entry.packaged_name}{suffix}"
        shutil.copy2(src, dest)
        os.chmod(dest, 0o755)
        copied.append(dest)
    return copied


def cleanup_paths(paths: list[Path], stage_dir: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
    if stage_dir.exists() and not any(stage_dir.iterdir()):
        stage_dir.rmdir()
    if stage_dir.name != "bin":
        parent = stage_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def npm_pack(package_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["npm", "pack", "--silent"],
        cwd=package_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("npm pack did not produce an output filename")
    tarball = lines[-1]
    pack_path = package_dir / tarball
    if not pack_path.exists():
        raise FileNotFoundError(f"npm pack output not found: {pack_path}")
    target = out_dir / tarball
    pack_path.replace(target)
    return target


def add_label(tarball: Path, label: str) -> Path:
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    normalized = label.strip().lower()
    if not normalized:
        raise ValueError("label cannot be empty")
    if any(ch not in allowed for ch in normalized):
        raise ValueError("label must be alphanumeric with '-' or '_'")
    renamed = tarball.with_name(f"{tarball.stem}-{normalized}{tarball.suffix}")
    tarball.replace(renamed)
    return renamed


def normalize_label(label: str) -> str:
    if not label:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    normalized = label.strip().lower()
    if not normalized:
        return ""
    if any(ch not in allowed for ch in normalized):
        raise ValueError("label must be alphanumeric with '-' or '_'")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build npm bundle with native binaries.")
    parser.add_argument(
        "--out-dir",
        default="target/npm",
        help="Directory to write the npm bundle into.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label suffix for the bundle filename.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    js_dir = repo_root / "js"
    out_dir = (repo_root / args.out_dir).resolve()
    label = normalize_label(args.label)

    print("Building Rust binaries via cargo ...")
    run(["cargo", "build", "--release", "--bins"], cwd=repo_root)

    stage_dir = js_dir / "bin"
    if label:
        stage_dir = stage_dir / label
    print(f"Staging binaries in {stage_dir} ...")
    staged = copy_binaries(repo_root, stage_dir)

    try:
        print("Building JS package via npm ...")
        run(["npm", "run", "build"], cwd=js_dir)
        print("Packing npm bundle ...")
        tarball = npm_pack(js_dir, out_dir)
        if label:
            tarball = add_label(tarball, label)
        print(f"Bundle written to {tarball}")
    finally:
        print("Cleaning staged binaries ...")
        cleanup_paths(staged, stage_dir)


if __name__ == "__main__":
    main()
