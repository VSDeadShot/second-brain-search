"""Command line entry point. Read-only in slice 1."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import click

from .config import ConfigError, load_config
from .discovery import DiscoveredDoc, discover_documents


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024 or unit == "MB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} MB"  # pragma: no cover


def _as_dict(doc: DiscoveredDoc) -> dict[str, object]:
    return {
        "path": str(doc.path),
        "project": doc.project,
        "project_root": str(doc.project_root),
        "rel_path": doc.rel_path,
        "size_bytes": doc.size_bytes,
        "mtime": doc.mtime,
    }


@click.group()
@click.version_option(package_name="second-brain-search")
def cli() -> None:
    """Semantic search over your own project documentation."""


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Emit the raw document list as JSON.")
def discover(as_json: bool) -> None:
    """List the documentation files that would be indexed."""
    try:
        config = load_config()
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    docs = discover_documents(config)

    if as_json:
        click.echo(json.dumps([_as_dict(d) for d in docs], indent=2))
        return

    if not docs:
        click.echo(f"No documents found under {config.scan_root}.")
        click.echo("Check include_patterns / exclude_paths in config.toml.")
        sys.exit(1)

    grouped: dict[str, list[DiscoveredDoc]] = defaultdict(list)
    for doc in docs:
        grouped[doc.project].append(doc)

    click.echo(f"Scanning {config.scan_root}\n")
    for project in sorted(grouped, key=str.lower):
        items = grouped[project]
        total = sum(d.size_bytes for d in items)
        click.echo(f"  {project}  ({len(items)} files, {_human_bytes(total)})")
        for doc in items:
            click.echo(f"      {doc.rel_path}")
        click.echo("")

    click.echo(
        f"{len(docs)} documents across {len(grouped)} projects, "
        f"{_human_bytes(sum(d.size_bytes for d in docs))} total."
    )
