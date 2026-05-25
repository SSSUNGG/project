import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StepTiming:
    sections: Dict[str, float] = field(default_factory=dict)
    reasoning_called: bool = False
    risk_score: float = 0.0
    primitive: str = ""

    @property
    def total_ms(self) -> float:
        return sum(self.sections.values()) * 1000.0

    def section_ms(self, name: str) -> float:
        return self.sections.get(name, 0.0) * 1000.0


class Timer:
    def __init__(self):
        self._sections: Dict[str, float] = {}
        self._start = None

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        yield
        self._sections[name] = time.perf_counter() - t0

    def dump(self, reasoning: bool = False, risk: float = 0.0, primitive: str = "") -> StepTiming:
        return StepTiming(
            sections=dict(self._sections),
            reasoning_called=reasoning,
            risk_score=risk,
            primitive=primitive,
        )
