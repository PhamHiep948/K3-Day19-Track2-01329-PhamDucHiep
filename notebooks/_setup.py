"""Path bootstrap for lab notebooks.

Resolves the repo root (where `app/`, `scripts/`, `data/` live) regardless of
where Jupyter was launched from. Used by all 4 notebooks:

    import _setup  # noqa: F401   -- adds repo root to sys.path

Why: `sys.path.insert(0, "../scripts")` is cwd-relative and silently breaks
when the notebook runs from CI or a different working directory. `__file__`
is stable; cwd is not.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Jupyter + pyzmq on Windows uses ProactorEventLoop by default; zmq needs Selector.
if sys.platform == "win32":
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Windows: reconfigure stdout so Vietnamese / arrows in notebook prints don't crash.
try:
    from app.stdio_utf8 import ensure_utf8  # noqa: E402
    ensure_utf8()
except Exception:
    pass

# Notebooks shell out to the `feast` CLI. Under `make lab` the venv is already
# active, but under nbconvert / CI it is not, and the call dies with
# FileNotFoundError: 'feast'. Put the running interpreter's bin dir on PATH so
# the CLI resolves the same way in every execution context.
_BIN = Path(sys.executable).parent
if str(_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
