"""``lara`` command line entry point.

Split out of a single 2,500-line module. Each submodule owns one area of the tool and
registers its commands on the shared :data:`app` from :mod:`lara.cli._base`; importing
them here is what puts those commands on the application.

**Commands import lara modules inside the function body, not at the top.** ``lara --help``
must not pay for torch, and it stays under a second rather than fifteen because nothing
heavy is imported until a command that needs it actually runs. In-function imports of
*stdlib* modules have no such excuse and are merely noise.
"""

from lara.cli._base import app, console

# isort: off
# This order is the order `lara --help` prints the commands in: Typer keeps registration
# order rather than sorting, so it is arranged as the work runs -- ready the machine,
# build a corpus, search it, fine-tune against it, serve it. Sorting these imports would
# silently reorder the help.
from lara.cli import setup      # noqa: F401 — imported for its command registrations
from lara.cli import corpus     # noqa: F401
from lara.cli import search     # noqa: F401
from lara.cli import finetune   # noqa: F401
from lara.cli import serve      # noqa: F401
from lara.cli import settings   # noqa: F401
from lara.cli import scope      # noqa: F401
from lara.cli import dataset    # noqa: F401
# isort: on

__all__ = ["app", "console"]
