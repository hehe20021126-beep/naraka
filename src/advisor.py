from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

from .types import ActionState, Detection, FrameState


@dataclass
class AdvisorConfig:
    close_distance: float = 150.0
    mid_distance: float = 320.0
    low_hp_threshold: float = 0.35


class CombatAdvisor:
    def __init__(self, cfg: AdvisorConfig | None = None):
        self.cfg = cfg or AdvisorConfig()

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    def build_state(self, detections: Iterable[Detection], self_hp: float = 0.8) -> FrameState:
        self_pos = None
        enemies: list[Tuple[int, int]] = []
        enemy_action = ActionState.IDLE

        for d in detections:
            c = self._center(d.bbox)
            if d.label == "self":
                self_pos = c
            elif d.label == "enemy":
                enemies.append(c)
                if d.action in (ActionState.CHARGE_ATTACK, ActionState.PARRY, ActionState.LIGHT_ATTACK):
                    enemy_action = d.action

        nearest = None
        if self_pos and enemies:
            nearest = min(math.dist(self_pos, e) for e in enemies)

        return FrameState(
            self_pos=self_pos,
            enemy_positions=enemies,
            nearest_enemy_distance=nearest,
            enemy_action=enemy_action,
            self_hp=self_hp,
        )

    def suggest(self, state: FrameState) -> str:
        if state.self_pos is None:
            return "未识别到己方角色，先调整视角保持目标在画面中央。"

        if not state.enemy_positions:
            return "未识别到敌人：优先听音辨位+滑步找人，避免空放蓄力。"

        d = state.nearest_enemy_distance
        if d is None:
            return "目标距离异常，建议先拉开视角重新锁定。"

        if state.self_hp < self.cfg.low_hp_threshold:
            if d < self.cfg.close_distance:
                return "血量偏低且贴身：先闪避/钩锁脱离，再回头反抓后摇。"
            return "血量偏低：保持中距离拉扯，诱导对手先手再反打。"

        if state.enemy_action == ActionState.CHARGE_ATTACK:
            if d < self.cfg.close_distance:
                return "敌方疑似蓄力近身：优先振刀或侧闪，抓硬直反击。"
            return "敌方在蓄力：可假动作逼振，再用平击起手。"

        if state.enemy_action == ActionState.PARRY:
            return "敌方疑似振刀动作：停顿半拍，改用骗振或抓投。"

        if d < self.cfg.close_distance:
            return "近距离缠斗：平A一段后注意停手，防对手振刀反制。"

        if d < self.cfg.mid_distance:
            return "中距离博弈：用滑步平A试探，保留闪避应对反手蓄力。"

        return "远距离：先走位逼墙角，利用地形限制对手闪避路线。"
