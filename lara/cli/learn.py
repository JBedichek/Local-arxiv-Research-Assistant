"""``lara learn`` -- courses built from the corpus."""

from __future__ import annotations

from pathlib import Path

import typer

from lara.cli._base import app, console

learn_app = typer.Typer(help="Learn -- courses built from the corpus")
app.add_typer(learn_app, name="learn")


@learn_app.command("import-courses")
def import_courses(
    source: Path = typer.Argument(..., help="a directory of courses, e.g. ~/.autoresearch/courses"),
) -> None:
    """Copy courses from another store into lara's (~/.lara/courses). Never overwrites."""
    from lara.learn import migrate

    done = migrate.copy_courses(source.expanduser())
    console.print(f"copied {len(done['copied'])} course(s); {len(done['skipped'])} already present")
    for name in done["copied"]:
        console.print(f"  + {name}")
