"""Force UTF-8 on Windows consoles.

Git Bash / cmd on Windows default to cp1252. Printing `→`, `•`, or Vietnamese
then raises UnicodeEncodeError and aborts seed / verify / notebooks. Import
this module as early as possible (side-effect: reconfigures stdout/stderr).
"""
from __future__ import annotations

import sys


def ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is None:
            continue
        try:
            reconf(encoding="utf-8", errors="replace")
        except Exception:
            pass


ensure_utf8()
