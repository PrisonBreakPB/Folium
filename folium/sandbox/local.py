"""Local command execution backend."""

import locale
import os
import re
import signal
import subprocess


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


class LocalCommandExecutor:
    def __init__(self):
        self.cwd: str | None = None

    def run(self, command: str, timeout: int = 120) -> str:
        warning = check_dangerous(command)
        if warning:
            return f"[Warning] Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."

        cwd = self.cwd or os.getcwd()

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
                terminate_process_tree(proc)
                stdout_bytes, stderr_bytes = proc.communicate()
                stdout = decode_process_output(stdout_bytes)
                stderr = decode_process_output(stderr_bytes)
                out = stdout or ""
                if stderr:
                    out += f"\n[stderr]\n{stderr}"
                out += f"\nError: timed out after {timeout}s and terminated process"
                return truncate_output(out)

            stdout = decode_process_output(stdout_bytes)
            stderr = decode_process_output(stderr_bytes)

            if proc.returncode == 0:
                self._update_cwd(command, cwd)
            out = stdout
            if stderr:
                out += f"\n[stderr]\n{stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"
            return truncate_output(out)
        except Exception as e:
            return f"Error running command: {e}"

    def _update_cwd(self, command: str, current_cwd: str):
        parts = command.split("&&")
        for part in parts:
            part = part.strip()
            if part.startswith("cd "):
                target = part[3:].strip().strip("'\"")
                if target:
                    new_dir = os.path.normpath(os.path.join(current_cwd, os.path.expanduser(target)))
                    if os.path.isdir(new_dir):
                        self.cwd = new_dir


def check_dangerous(cmd: str) -> str | None:
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return reason
    return None


def terminate_process_tree(proc: subprocess.Popen):
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


def decode_process_output(data: bytes | str | None) -> str:
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


def truncate_output(out: str) -> str:
    if len(out) > 15_000:
        out = (
            out[:6000]
            + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
            + out[-3000:]
        )
    return out.strip() or "(no output)"
