#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Stitch per-platform npm bundles into a unified package."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def parse_sources(values: list[str]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH syntax, got '{value}'")
        label, path = value.split("=", 1)
        label = label.strip().lower()
        if not label:
            raise ValueError("label cannot be empty")
        if any(ch not in allowed for ch in label):
            raise ValueError("label must be alphanumeric with '-' or '_'")
        sources.append((label, Path(path).expanduser()))
    return sources


def discover_bundles(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"bundle source not found: {path}")
    bundles = sorted(p for p in path.rglob("*.tgz") if p.is_file())
    if not bundles:
        raise FileNotFoundError(f"no bundles found under {path}")
    return bundles


def extract_bundle(bundle: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, mode="r:gz") as archive:
        archive.extractall(dest)
    package_root = dest / "package"
    if not package_root.exists():
        raise FileNotFoundError(f"bundle missing package directory: {bundle}")
    return package_root


def read_package_json(package_root: Path) -> dict[str, str]:
    package_path = package_root / "package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"missing package.json: {package_path}")
    return json.loads(package_path.read_text())


def ensure_label_bin(package_root: Path, label: str) -> None:
    target = package_root / "bin" / label
    if not target.exists():
        raise FileNotFoundError(f"missing bin directory for label '{label}'")


def copy_label_bin(source_root: Path, dest_root: Path, label: str) -> None:
    source_bin = source_root / "bin" / label
    if not source_bin.exists():
        raise FileNotFoundError(f"missing bin directory for label '{label}'")
    dest_bin = dest_root / "bin" / label
    if dest_bin.exists():
        raise FileExistsError(f"bin directory already exists: {dest_bin}")
    dest_bin.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_bin, dest_bin)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect npm bundle artifacts into a unified directory."
    )
    parser.add_argument(
        "--output",
        default="target/npm-release",
        help="Directory to write the bundled artifacts into.",
    )
    parser.add_argument("sources", nargs="+", help="Bundle sources as LABEL=PATH.")
    args = parser.parse_args()

    sources = parse_sources(args.sources)
    output = Path(args.output).resolve()

    first_label, first_path = sources[0]
    bundles = discover_bundles(first_path)
    if len(bundles) != 1:
        raise RuntimeError(f"expected 1 bundle for '{first_label}', got {len(bundles)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        base_root = extract_bundle(bundles[0], temp_root / f"base-{first_label}")
        base_meta = read_package_json(base_root)
        ensure_label_bin(base_root, first_label)

        for label, source_path in sources[1:]:
            bundles = discover_bundles(source_path)
            if len(bundles) != 1:
                raise RuntimeError(f"expected 1 bundle for '{label}', got {len(bundles)}")
            extract_root = extract_bundle(bundles[0], temp_root / f"src-{label}")
            meta = read_package_json(extract_root)
            if meta.get("name") != base_meta.get("name") or meta.get("version") != base_meta.get(
                "version"
            ):
                raise RuntimeError(
                    f"bundle metadata mismatch for '{label}': "
                    f"{meta.get('name')}@{meta.get('version')}"
                )
            copy_label_bin(extract_root, base_root, label)

        output.mkdir(parents=True, exist_ok=True)
        tarball = npm_pack(base_root, output)
        print(f"Stitched bundle written to {tarball}")


if __name__ == "__main__":
    main()
