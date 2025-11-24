#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["click>=8", "rich>=13"]
# ///
"""Collect per-platform wheel artifacts and a source distribution into a unified directory."""

import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

import click
from rich.console import Console

console = Console()


class WheelSource(click.ParamType):
    name = "label=path"

    def convert(self, value: str, param, ctx):  # type: ignore[override]
        if "=" not in value:
            self.fail("expected LABEL=PATH syntax", param, ctx)
        label, path = value.split("=", 1)
        label = label.strip().lower()
        if not label:
            self.fail("label cannot be empty", param, ctx)
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
        if any(ch not in allowed for ch in label):
            self.fail("label must be alphanumeric with '-' or '_'", param, ctx)
        return label, Path(path).expanduser()


def _discover_wheels(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"wheel source not found: {path}")
    return sorted(p for p in path.rglob("*.whl") if p.is_file())


def _discover_sdist(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"sdist source not found: {path}")
    sdists = sorted(p for p in path.rglob("*.tar.gz") if p.is_file())
    if not sdists:
        raise FileNotFoundError(f"no sdist found under {path}")
    return sdists[0]


def _log_wheel_contents(wheel: Path) -> None:
    console.rule(f"Wheel contents: {wheel.name}")
    with zipfile.ZipFile(wheel) as archive:
        for name in sorted(archive.namelist()):
            console.print(f"[cyan]{name}[/cyan]")


def _log_sdist_contents(sdist: Path) -> None:
    console.rule(f"sdist contents: {sdist.name}")
    with tarfile.open(sdist, mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            console.print(f"[magenta]{member.name}[/magenta]")


@click.command()
@click.option("--output", "output_dir", default="target/release-wheels", show_default=True)
@click.option(
    "--sdist",
    "sdist_path",
    required=True,
    type=click.Path(exists=False, path_type=Path),
    help="Path to the built sdist file or directory containing it.",
)
@click.argument("sources", nargs=-1, type=WheelSource())
def main(output_dir: str, sdist_path: Path, sources: Iterable[tuple[str, Path]]) -> None:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not sources:
        raise click.UsageError("provide at least one LABEL=PATH source")

    collected: list[Path] = []
    for label, source_path in sources:
        wheels = _discover_wheels(source_path)
        if not wheels:
            raise click.ClickException(f"no wheels found under {source_path}")
        console.log(f"[green]Collecting {len(wheels)} wheel(s) from '{label}' into {output}")
        for wheel in wheels:
            destination = output / wheel.name
            shutil.copy2(wheel, destination)
            console.log(f"[blue]Added -> {destination}")
            collected.append(destination)
    sdist = _discover_sdist(sdist_path)
    sdist_dest = output / sdist.name
    shutil.copy2(sdist, sdist_dest)
    console.log(f"[magenta]Added sdist -> {sdist_dest}")
    for wheel in collected:
        _log_wheel_contents(wheel)
    _log_sdist_contents(sdist_dest)


if __name__ == "__main__":
    main()
