"""Docker-backed command execution backend."""

import os
import shutil
import subprocess
import uuid
from pathlib import Path

from ..observability.context import record_sandbox_event
from .local import check_dangerous, decode_process_output, truncate_output
from .session import get_current_session, get_host_workspace, use_copy_workspace


class DockerSandboxExecutor:
    def __init__(
        self,
        image: str = "python:3.11-slim",
        workspace: str | None = None,
        network: str = "bridge",
        cpus: str = "1",
        memory: str = "2g",
        pids_limit: int = 256,
    ):
        self.image = image
        if workspace is None:
            self.session = get_current_session() if use_copy_workspace() else None
            self.workspace = (
                self.session.workspace.resolve()
                if self.session is not None
                else get_host_workspace()
            )
        else:
            self.session = None
            self.workspace = Path(workspace).resolve()
        self.network = network
        self.cpus = cpus
        self.memory = memory
        self.pids_limit = pids_limit
        self.container_name = f"folium-sandbox-{uuid.uuid4().hex[:12]}"
        self.container_id: str | None = None

    def run(self, command: str, timeout: int = 120) -> str:
        warning = check_dangerous(command)
        if warning:
            self._record_event("command_blocked", {
                "reason": warning,
                "command_hash": _stable_command_hash(command),
            })
            return f"[Warning] Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."
        if not shutil.which("docker"):
            self._record_event("docker_missing", {})
            return "Error: Docker sandbox requested but docker executable was not found."

        try:
            self._ensure_container()
            proc = subprocess.Popen(
                [
                    "docker",
                    "exec",
                    "-w",
                    "/workspace",
                    self.container_id,
                    "sh",
                    "-lc",
                    command,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_bytes, stderr_bytes = proc.communicate()
                self.cleanup()
                self._record_event("command_timeout", {
                    "timeout": timeout,
                    "command_hash": _stable_command_hash(command),
                })
                stdout = decode_process_output(stdout_bytes)
                stderr = decode_process_output(stderr_bytes)
                out = stdout or ""
                if stderr:
                    out += f"\n[stderr]\n{stderr}"
                out += f"\nError: timed out after {timeout}s and terminated process"
                return truncate_output(out)

            stdout = decode_process_output(stdout_bytes)
            stderr = decode_process_output(stderr_bytes)
            out = stdout
            if stderr:
                out += f"\n[stderr]\n{stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"
            self._record_event("command_finished", {
                "returncode": proc.returncode,
                "timeout": timeout,
                "command_hash": _stable_command_hash(command),
            })
            return truncate_output(out)
        except Exception as e:
            self._record_event("command_error", {
                "error": str(e),
                "command_hash": _stable_command_hash(command),
            })
            return f"Error running command in Docker sandbox: {e}"

    def cleanup(self):
        if not self.container_id:
            return
        container_id = self.container_id
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._record_event("container_removed", {
            "container_id": container_id,
        })
        self.container_id = None

    def _ensure_container(self):
        if self.container_id:
            return

        self.workspace.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "--network",
                self.network,
                "--cpus",
                self.cpus,
                "--memory",
                self.memory,
                "--pids-limit",
                str(self.pids_limit),
                "-v",
                f"{self.workspace}:/workspace",
                "-w",
                "/workspace",
                self.image,
                "sleep",
                "infinity",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            stderr = decode_process_output(proc.stderr)
            raise RuntimeError(f"failed to start Docker sandbox: {stderr}")
        self.container_id = decode_process_output(proc.stdout).strip()
        self._record_event("container_started", {
            "container_id": self.container_id,
            "container_name": self.container_name,
            "image": self.image,
            "network": self.network,
            "cpus": self.cpus,
            "memory": self.memory,
            "pids_limit": self.pids_limit,
        })

    def _record_event(self, action: str, metadata: dict):
        base = {
            "container_id": self.container_id,
            "container_name": self.container_name,
            "workspace": str(self.workspace),
            "workspace_mode": "copy" if self.session is not None else "host",
        }
        if self.session is not None:
            base["sandbox_session_id"] = self.session.session_id
            base["host_workspace"] = str(self.session.host_workspace)
        base.update(metadata)
        record_sandbox_event(action, base)


def _stable_command_hash(command: str) -> str:
    import hashlib

    return hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()
