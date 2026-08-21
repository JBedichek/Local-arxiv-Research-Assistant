"""Support ``python -m lara.cli``.

A package needs this: the ``if __name__ == "__main__"`` guard that worked in the
single-module version never fires for ``__init__.py`` under ``-m``.
"""

from lara.cli import app

app()
