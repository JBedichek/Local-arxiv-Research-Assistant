"""Arrow-key selection, with a typed fallback wherever that is not possible.

Deliberately stdlib-only. The obvious way to get a menu is ``questionary``, which
drags in ``prompt_toolkit`` — several MB for one prompt, in a base install whose whole
premise is that it contains only what it takes to search and read. Reading four key
codes is not worth a dependency of that size.

The terminal is put in raw mode **only for the duration of a single keypress**.
Rendering happens in cooked mode, so ``rich`` emits ordinary newlines and nothing
staircases; the alternative — holding raw mode across the redraw — means translating
every ``\\n`` to ``\\r\\n`` by hand.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

UP, DOWN, ENTER, ESCAPE, OTHER = "up", "down", "enter", "escape", "other"


def interactive(stream=None) -> bool:
    """Whether a full-screen selector can be driven on this terminal.

    False for pipes, CI, ``TERM=dumb`` and anyone who sets ``LARA_NO_TUI``. Every
    caller must have a typed path for these cases — piping answers into ``lara setup``
    is a supported way to run it.
    """
    if os.environ.get("LARA_NO_TUI"):
        return False
    stream = stream or sys.stdin
    try:
        return bool(stream.isatty() and sys.stdout.isatty()
                    and os.environ.get("TERM", "") != "dumb")
    except (AttributeError, ValueError):
        return False


def _read_key_posix() -> str:
    import select as _select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # os.read, not sys.stdin.read: sys.stdin is buffered, so reading the ESC would
        # pull the rest of the sequence into Python's buffer where select() cannot see
        # it. The poll below then reports "nothing pending" and every arrow key is
        # misread as a bare escape.
        ch = os.read(fd, 1)
        if ch == b"\x03":
            raise KeyboardInterrupt
        if ch in (b"\r", b"\n"):
            return ENTER
        if ch != b"\x1b":
            return OTHER
        # CSI sequences arrive as one burst. A lone ESC has nothing behind it, so a
        # short poll distinguishes "escape" from "arrow" without blocking on either.
        rest = b""
        while len(rest) < 2 and _select.select([fd], [], [], 0.05)[0]:
            more = os.read(fd, 2 - len(rest))
            if not more:
                break
            rest += more
        if not rest:
            return ESCAPE
        return {b"[A": UP, b"[B": DOWN, b"OA": UP, b"OB": DOWN}.get(rest, OTHER)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _read_key_windows() -> str:
    import msvcrt

    ch = msvcrt.getwch()
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch == "\r":
        return ENTER
    if ch == "\x1b":
        return ESCAPE
    if ch in ("\x00", "\xe0"):        # arrows are a two-character sequence here
        return {"H": UP, "P": DOWN}.get(msvcrt.getwch(), OTHER)
    return OTHER


def read_key() -> str:
    """One keypress, normalised to UP / DOWN / ENTER / ESCAPE / OTHER."""
    if os.name == "nt":
        return _read_key_windows()
    return _read_key_posix()


def select(count: int, render: Callable[[int], object], *, console,
           initial: int = 0) -> int | None:
    """Move a cursor over ``count`` rows and return the chosen index.

    ``render(cursor)`` returns whatever rich should draw for that cursor position.
    Returns ``None`` if the user pressed escape, which callers should read as "keep
    what was already selected" rather than as an error.
    """
    from rich.live import Live

    if count <= 0:
        return None
    cursor = min(max(initial, 0), count - 1)

    with Live(render(cursor), console=console, auto_refresh=False,
              transient=False) as live:
        while True:
            key = read_key()
            if key == ENTER:
                return cursor
            if key == ESCAPE:
                return None
            if key == UP:
                cursor = (cursor - 1) % count
            elif key == DOWN:
                cursor = (cursor + 1) % count
            else:
                continue
            live.update(render(cursor))
            live.refresh()
