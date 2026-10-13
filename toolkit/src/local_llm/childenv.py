"""The environment that `lms` is started with.

"Environment variables" are the named settings every program inherits from whatever
started it. A child process gets a copy of its parent's unless told otherwise, which is
out of sight, and that is how one variable broke the model server without a trace.

## The variable, and what it does

LM Studio's app is built on Electron, a framework that wraps a web page and a copy of the
Node.js script engine into a desktop program. Electron reads one switch from its
environment:

    ELECTRON_RUN_AS_NODE=1   ->   "do not open the app; behave as the bare script engine"

Editors built on Electron, VS Code among them, set that switch for the helper programs
they start, and every terminal, assistant or task runner inside the editor inherits it.
When `lms` then tries to wake LM Studio's background service, it starts the app's own
program file, the switch is inherited again, and the app does not start as an app at
all. It reads its service flag as a script-engine option, rejects it and exits:

    Bionic.exe: bad option: --run-as-service

Nothing reports this. `lms` only sees that no service appeared:

    Waking up LM Studio service...
    Error: Timed out waiting for LM Studio daemon to start.

Measured on 13 October 2026: with the switch set, `lms server start` from a stopped
service failed after 60 seconds on every one of five tries and no app process ever
appeared. With the switch removed, the same command succeeded in 7 seconds. A service
that is already running hides the fault completely, which is why it looked like a
problem that came and went.

## What this module does about it

Every place in the toolkit that starts `lms` passes `lms_environment()` as the child's
environment, so the switch never reaches LM Studio whatever the toolkit itself was
started from. Removing it is safe everywhere: nothing the toolkit starts wants an
Electron app to run as a script engine.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# The one switch that makes an Electron app run as a bare script engine (see above).
_ELECTRON_SWITCH = "ELECTRON_RUN_AS_NODE"


def lms_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment to start `lms` with: the current one without Electron's switch.

        base = {"PATH": "...", "ELECTRON_RUN_AS_NODE": "1", "USERPROFILE": "..."}
          ->   {"PATH": "...", "USERPROFILE": "..."}

        base = {"PATH": "..."}   ->   {"PATH": "..."}     (nothing to remove)

    `base` is passed in by tests so they can show exactly what is removed without
    changing the real environment. Left out, the process's own environment is used,
    read at the moment of the call and not when this module was imported, so a variable
    set later is still seen.

    A new dict is always returned and `base` is never changed: the caller's environment
    must stay as it was, because the editor that set the switch still needs it for its
    own helper programs.
    """
    source = os.environ if base is None else base
    return {name: value for name, value in source.items() if name != _ELECTRON_SWITCH}


__all__ = ["lms_environment"]
