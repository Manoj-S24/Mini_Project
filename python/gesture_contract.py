from dataclasses import dataclass
import math
from typing import Protocol, Any


@dataclass(frozen=True)
class GestureState:
    """Confirmed gesture results for one camera frame."""

    left_pinch_active: bool = False
    right_click_active: bool = False
    scroll_active: bool = False

    # Positive: hand moves down. Negative: hand moves up.
    scroll_delta_y: float = 0.0

    # Gesture confidence, not Hand.score.
    confidence: float = 1.0

    label: str = "NEUTRAL"
    debug_message: str = ""

    def __post_init__(self):
        for name in (
            "left_pinch_active",
            "right_click_active",
            "scroll_active",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

        if not math.isfinite(self.scroll_delta_y):
            raise ValueError("scroll_delta_y must be finite")

        if not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if not isinstance(self.label, str):
            raise TypeError("label must be a string")

        if not isinstance(self.debug_message, str):
            raise TypeError("debug_message must be a string")


class GestureRecognizerProtocol(Protocol):
    """
    Interface expected from the gesture-recognition module.

    update() must be called on every camera frame, including when
    hand=None, and must return a GestureState.
    """

    def update(
        self,
        hand: Any | None,
        timestamp: float,
    ) -> GestureState:
        ...