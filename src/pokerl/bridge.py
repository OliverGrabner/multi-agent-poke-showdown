"""Client for the Node bridge (bridge/bridge.js) that hosts Showdown battles."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

BRIDGE_DIR = Path(os.environ.get("POKERL_BRIDGE_DIR", Path(__file__).resolve().parents[2] / "bridge"))


class BridgeError(RuntimeError):
    """The bridge rejected a command or the simulator raised."""


class Bridge:
    """One Node process; it can hold many battles at once. Calls are serialized by a lock."""

    def __init__(self, node: str | None = None):
        node = node or os.environ.get("POKERL_NODE") or shutil.which("node")
        if not node:
            raise BridgeError("Node.js not found; install it or set POKERL_NODE")
        script = BRIDGE_DIR / "bridge.js"
        if not (BRIDGE_DIR / "node_modules" / "pokemon-showdown").is_dir():
            raise BridgeError(f"Run `npm ci` in {BRIDGE_DIR} first")
        self._process = subprocess.Popen(
            [node, str(script)],
            cwd=BRIDGE_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr is inherited: a full, undrained pipe would block the bridge.
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._next_id = 0

    def call(self, cmd: str, **args: Any) -> Any:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            try:
                self._process.stdin.write(json.dumps({"id": request_id, "cmd": cmd, "args": args}) + "\n")
                self._process.stdin.flush()
                line = self._process.stdout.readline()
            except (BrokenPipeError, OSError) as err:
                raise BridgeError("Bridge process died; see its stderr above") from err
            if not line:
                raise BridgeError(f"Bridge process exited with code {self._process.poll()}")
            response = json.loads(line)
            if response["id"] != request_id:
                raise BridgeError(f"Out-of-order response {response['id']} for request {request_id}")
            if not response["ok"]:
                raise BridgeError(f"{cmd}: {response['error']}\n{response.get('stack', '')}")
            return response["result"]

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def __enter__(self) -> Bridge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
