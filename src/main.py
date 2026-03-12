from __future__ import annotations

import argparse

import cv2
import mss
import numpy as np

from .advisor import CombatAdvisor
from .detector import DummyDetector


def frame_from_screen() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def render(frame: np.ndarray, detections, advice: str) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        color = (0, 255, 0) if d.label == "self" else (0, 120, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{d.label}:{d.action.value} {d.confidence:.2f}"
        cv2.putText(out, text, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.rectangle(out, (10, 10), (980, 60), (0, 0, 0), -1)
    cv2.putText(out, f"建议: {advice}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["camera", "screen", "video"], default="camera")
    p.add_argument("--video_path", default="")
    p.add_argument("--show", action="store_true")
    p.add_argument("--self_hp", type=float, default=0.8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detector = DummyDetector()
    advisor = CombatAdvisor()

    cap = None
    if args.source == "camera":
        cap = cv2.VideoCapture(0)
    elif args.source == "video":
        if not args.video_path:
            raise ValueError("source=video 时必须提供 --video_path")
        cap = cv2.VideoCapture(args.video_path)

    while True:
        if args.source == "screen":
            frame = frame_from_screen()
        else:
            ok, frame = cap.read()
            if not ok:
                break

        detections = detector.infer(frame)
        state = advisor.build_state(detections, self_hp=max(0.0, min(1.0, args.self_hp)))
        advice = advisor.suggest(state)
        vis = render(frame, detections, advice)

        if args.show:
            cv2.imshow("naraka-advisor", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
