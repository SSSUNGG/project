"""JSON Lines structured logging for experiment events."""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def get_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


class EventLogger:
    """Structured JSON-Lines event logger.

    Writes one JSON object per line to a .jsonl file.
    Event types: episode_start, episode_end, reasoner_call, primitive_start,
                 primitive_end, policy_step, error.
    """

    def __init__(
        self,
        log_path: str,
        phase: int,
        condition: str,
        task: str,
        console: bool = True,
    ):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.condition = condition
        self.task = task
        self.console = console
        self._commit_sha = get_commit_sha()
        self._logger = logging.getLogger(__name__)

    def log(self, event: str, **kwargs) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "condition": self.condition,
            "task": self.task,
            "event": event,
            "commit_sha": self._commit_sha,
            **kwargs,
        }
        line = json.dumps(record)
        with open(self.path, "a") as f:
            f.write(line + "\n")

        if self.console:
            self._logger.info("%s", line)

    def episode_start(self, seed: int, episode: int) -> None:
        self.log("episode_start", seed=seed, episode=episode)

    def episode_end(self, seed: int, episode: int, success: bool, wall_s: float) -> None:
        self.log("episode_end", seed=seed, episode=episode, success=success, wall_s=wall_s)

    def reasoner_call(self, seed: int, episode: int, step: int, risk: float, primitive: str, latency_ms: float) -> None:
        self.log("reasoner_call", seed=seed, episode=episode, step=step,
                 risk_score=risk, primitive=primitive, latency_ms=latency_ms)

    def error(self, seed: int, episode: int, step: int, msg: str) -> None:
        self.log("error", seed=seed, episode=episode, step=step, message=msg)
