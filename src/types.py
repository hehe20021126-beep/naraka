from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class ActionState(str, Enum):
    IDLE = "idle"
    LIGHT_ATTACK = "light_attack"
    CHARGE_ATTACK = "charge_attack"
    PARRY = "parry"
    DODGE = "dodge"


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    action: ActionState = ActionState.IDLE


@dataclass
class FrameState:
    self_pos: Tuple[int, int] | None
    enemy_positions: List[Tuple[int, int]]
    nearest_enemy_distance: float | None
    enemy_action: ActionState
    self_hp: float  # 0~1
