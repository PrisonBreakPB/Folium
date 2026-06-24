"""Shell command execution with safety checks.

Claude Code's BashTool is 1,143 lines. This is the distilled version:
- Output capture with truncation (head+tail preserved)
- Timeout support
- Dangerous command detection
- Working directory tracking (cd awareness)
"""

import os
import re
import signal
import locale
import subprocess
from .base import Tool

# track cwd across commands (Claude Code does this too)
_cwd: str | None = None

# patterns that could wreck the filesystem or leak secrets
_DANGEROUS_PATTERNS = [
    (r"\brm\s+(-\w*)?-r\w*\s+(/|~|\$HOME)", "recursive delete on home/root"),
    (r"\brm\s+(-\w*)?-rf\s", "force recursive delete"),
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*of=/dev/", "raw disk write"),
    (r">\s*/dev/sd[a-z]", "overwrite block device"),
    (r"\bchmod\s+(-R\s+)?777\s+/", "chmod 777 on root"),
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\bcurl\b.*\|\s*(sudo\s+)?bash", "pipe curl to bash"),
    (r"\bwget\b.*\|\s*(sudo\s+)?bash", "pipe wget to bash"),
]


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command. Returns stdout, stderr, and exit code. "
        "Use this for running tests, installing packages, git operations, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str, timeout: int = 120) -> str:
        global _cwd
        # safety check
        warning = _check_dangerous(command)
        if warning:
            return f"[Warning] Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."

        # use tracked working directory
        cwd = _cwd or os.getcwd()

        try:
            creationflags = 0
            start_new_session = os.name != "nt"
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                start_new_session=start_new_session,
                creationflags=creationflags,
            )
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(proc)
                stdout_bytes, stderr_bytes = proc.communicate()
                stdout = _decode_process_output(stdout_bytes)
                stderr = _decode_process_output(stderr_bytes)
                out = stdout or ""
                if stderr:
                    out += f"\n[stderr]\n{stderr}"
                out += f"\nError: timed out after {timeout}s and terminated process"
                return _truncate_output(out)
            stdout = _decode_process_output(stdout_bytes)
            stderr = _decode_process_output(stderr_bytes)

            # track cd commands so next command runs in the right place
            if proc.returncode == 0:
                _update_cwd(command, cwd)
            out = stdout
            if stderr:
                out += f"\n[stderr]\n{stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"
            return _truncate_output(out)
        except Exception as e:
            return f"Error running command: {e}"


def _check_dangerous(cmd: str) -> str | None:
    """Return a warning string if the command looks destructive, else None."""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return reason
    return None


def _update_cwd(command: str, current_cwd: str):
    """Track directory changes from cd commands."""
    global _cwd
    # simple heuristic: look for cd at the end of a && chain or standalone
    parts = command.split("&&")
    for part in parts:
        part = part.strip()
        if part.startswith("cd "):
            target = part[3:].strip().strip("'\"")
            if target:
                new_dir = os.path.normpath(os.path.join(current_cwd, os.path.expanduser(target)))
                if os.path.isdir(new_dir):
                    _cwd = new_dir


def _terminate_process_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            text=True,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


def _decode_process_output(data: bytes | str | None) -> str:
    if not data:
        return ""
    if isinstance(data, str):
        return data

    encodings = ["utf-8-sig", locale.getpreferredencoding(False), "gbk", "cp936"]
    tried = set()
    for encoding in encodings:
        if not encoding or encoding in tried:
            continue
        tried.add(encoding)
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _truncate_output(out: str) -> str:
    # keep head + tail to preserve the most useful info
    if len(out) > 15_000:
        out = (
            out[:6000]
            + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
            + out[-3000:]
        )
    return out.strip() or "(no output)"
