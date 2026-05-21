from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PrimitiveState:
    """Internal state carried across primitive steps."""
    phase: int = 0               # index into the primitive's phase sequence
    step_in_phase: int = 0       # how many steps completed in the current phase
    target_offset: Any = None    # phase-specific target (e.g., delta xyz)
    abort_episode: bool = False  # set by request_help
    extra: dict = field(default_factory=dict)


class BasePrimitive(ABC):
    """Abstract base for multi-step fallback primitives.

    A primitive runs independently of the VLA policy for up to its max_steps.
    The wrapper calls step() repeatedly until is_done=True, then resumes VLA.
    """

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Maximum number of steps before forced termination."""
        ...

    @abstractmethod
    def step(
        self, obs: dict, state: PrimitiveState
    ) -> tuple[Any, PrimitiveState, bool]:
        """Execute one step of the primitive.

        Args:
            obs:   Current normalized observation dict.
            state: Current primitive state.

        Returns:
            (action, next_state, is_done)
            - action:    np.ndarray (7,) float32 — 7-DoF EE delta pose
            - next_state: Updated PrimitiveState
            - is_done:   True means primitive is finished; VLA resumes next step
        """
        ...

    def initial_state(self) -> PrimitiveState:
        return PrimitiveState()
