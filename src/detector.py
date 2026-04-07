from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .types import ActionState, Detection


@dataclass
class DetectorConfig:
    self_label: str = "self"
    enemy_label: str = "enemy"


class BaseDetector:
    def infer(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class DummyDetector(BaseDetector):
    """
    占位检测器：
    - 使用亮度阈值做简单轮廓检测，模拟敌人框。
    - 屏幕中心附近假定为玩家自身。

    用于演示主流程，实际请替换为训练后的目标检测模型。
    """

    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()

    def infer(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        h, w = frame.shape[:2]

        # 玩家位置：默认屏幕中心
        center_w, center_h = w // 2, h // 2
        detections.append(
            Detection(
                label=self.cfg.self_label,
                confidence=0.6,
                bbox=(center_w - 20, center_h - 40, center_w + 20, center_h + 40),
                action=ActionState.IDLE,
            )
        )

        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < 600:
                continue

            # 规则模拟：高宽比较大时假设敌人蓄力，否则平击
            ratio = ch / max(cw, 1)
            enemy_action = ActionState.CHARGE_ATTACK if ratio > 1.8 else ActionState.LIGHT_ATTACK

            detections.append(
                Detection(
                    label=self.cfg.enemy_label,
                    confidence=0.5,
                    bbox=(x, y, x + cw, y + ch),
                    action=enemy_action,
                )
            )

        return detections
