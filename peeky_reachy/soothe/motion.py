"""Expressive comfort motion. Uses the Reachy SDK when present, else logs intent.

Identical SDK calls drive the MuJoCo sim and real hardware, so this module is
robot-agnostic; passing ``mini=None`` (e.g. on the dev laptop) makes every
routine a no-op that just logs what it would have done.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

log = logging.getLogger("peeky.motion")


class ComfortMotion:
    def __init__(self, mini=None):
        self.mini = mini
        self._head_pose = None
        if mini is not None:
            from reachy_mini.utils import create_head_pose

            self._head_pose = create_head_pose

    def _goto(self, *, head=None, antennas=None, body_yaw=None, duration=1.0, method="minjerk"):
        if self.mini is None:
            log.info("[no-robot] motion head=%s antennas=%s body_yaw=%s dur=%.2f",
                     head, antennas, body_yaw, duration)
            time.sleep(min(duration, 0.05))
            return
        kwargs = {"duration": duration, "method": method}
        if head is not None:
            kwargs["head"] = head
        if antennas is not None:
            kwargs["antennas"] = antennas
        if body_yaw is not None:
            kwargs["body_yaw"] = body_yaw
        self.mini.goto_target(**kwargs)

    def rest(self) -> None:
        head = self._head_pose() if self._head_pose else "rest"
        self._goto(head=head, antennas=[0.0, 0.0], duration=0.8)

    def idle_breathe(self) -> None:
        """One slow 'breathing' bob; call periodically while listening."""
        head = self._head_pose(z=6, mm=True, degrees=True) if self._head_pose else "breathe-up"
        self._goto(head=head, duration=1.6, method="ease_in_out")
        head = self._head_pose(z=0, mm=True, degrees=True) if self._head_pose else "breathe-down"
        self._goto(head=head, duration=1.6, method="ease_in_out")

    def attend(self, doa_rad: Optional[float]) -> None:
        """Turn gently toward a sound direction (radians from DoA)."""
        yaw_deg = 0.0 if doa_rad is None else max(-160, min(160, math.degrees(doa_rad) - 90))
        head = self._head_pose(yaw=yaw_deg, mm=True, degrees=True) if self._head_pose else f"attend yaw={yaw_deg:.0f}"
        self._goto(head=head, body_yaw=None, duration=0.8, method="minjerk")

    def comfort(self, cycles: int = 3) -> None:
        """A soft rocking + antenna 'wiggle' loop to accompany soothing audio."""
        for _ in range(cycles):
            head = self._head_pose(roll=8, z=4, mm=True, degrees=True) if self._head_pose else "rock-left"
            self._goto(head=head, antennas=[0.4, -0.4], duration=1.2, method="ease_in_out")
            head = self._head_pose(roll=-8, z=4, mm=True, degrees=True) if self._head_pose else "rock-right"
            self._goto(head=head, antennas=[-0.4, 0.4], duration=1.2, method="ease_in_out")
        self.rest()
