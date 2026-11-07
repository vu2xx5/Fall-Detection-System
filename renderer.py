"""Vẽ overlay kết quả lên khung hình video."""

from __future__ import annotations

import cv2
import numpy as np

from track_state import PersonTrackState

COLOR_FALLEN = (0, 0, 255)
COLOR_NORMAL = (0, 255, 0)
COLOR_COLLECTING = (0, 200, 255)
COLOR_HUD = (255, 255, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
LABEL_THICKNESS = 2
HUD_SCALE = 0.8
HUD_ORIGIN = (15, 35)
LABEL_MIN_Y = 22


def _label_for(track_id: int, state: PersonTrackState) -> tuple[str, tuple[int, int, int], int]:
    """Chọn nhãn, màu và độ dày nét vẽ theo trạng thái của một người."""
    if state.is_fallen:
        sec = state.seconds_fallen
        time_str = f" {sec:.0f}s" if sec is not None else ""
        return (
            f"ID #{track_id} | FALLEN!{time_str} ({state.fall_prob * 100:.0f}%)",
            COLOR_FALLEN,
            3,
        )

    if not state.is_ready:
        have, need = state.progress
        return (
            f"ID #{track_id} | Collecting {have}/{need}",
            COLOR_COLLECTING,
            2,
        )

    # Chưa bị kết luận ngã: hiển thị lý do để dễ gỡ lỗi báo giả.
    reason_label = state.reason.replace("_", " ").title()
    return (
        f"ID #{track_id} | {reason_label} {state.fall_prob * 100:.0f}%",
        COLOR_NORMAL,
        2,
    )


def draw_person(
    frame: np.ndarray,
    box: np.ndarray,
    track_id: int,
    state: PersonTrackState,
) -> None:
    """Vẽ bounding box và nhãn trạng thái cho một người."""
    x1, y1, x2, y2 = (int(value) for value in box)
    label, color, thickness = _label_for(track_id, state)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, max(LABEL_MIN_Y, y1 - 10)),
        FONT,
        LABEL_SCALE,
        color,
        LABEL_THICKNESS,
    )


def draw_hud(frame: np.ndarray, fps: float, people_count: int) -> None:
    """Vẽ FPS và số người đang theo dõi ở góc trên trái."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | People: {people_count}",
        HUD_ORIGIN,
        FONT,
        HUD_SCALE,
        COLOR_HUD,
        LABEL_THICKNESS,
    )
