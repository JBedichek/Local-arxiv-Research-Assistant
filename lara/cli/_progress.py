"""One primed progress generator, instead of ten hand-rolled ones.

Every long-running command reports the same way: the worker ``.send()``s a record per
batch, step or file, and the CLI prints a line for it. Written out by hand that is a
``while True: record = yield`` loop, plus the ``next(g)`` that primes it -- and the
priming is the part worth removing. It has to happen before the first ``send()`` and it
does nothing visible, so omitting it fails later and elsewhere, with a ``TypeError`` from
inside whichever worker was reporting.

Passing a function that turns one record into one line leaves nothing to forget.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

from lara.cli._base import console


def reporter(render: Callable[[Any], str | None]) -> Generator[None, Any, None]:
    """A generator, already primed, that prints ``render(record)`` for each record sent.

    ``render`` returns the line to print, or None to print nothing for that record --
    which is how a step that only matters sometimes (an early stop, an epoch boundary)
    stays a single expression rather than a branch in a loop.
    """
    def gen() -> Generator[None, Any, None]:
        while True:
            line = render((yield))
            if line is not None:
                console.print(line)

    g = gen()
    next(g)
    return g
