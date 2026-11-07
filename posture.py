"""Hình học tư thế tính từ 17 keypoint COCO của YOLOv8-Pose.

Lớp này độc lập hoàn toàn với mạng nơ-ron: nó chỉ trả lời câu hỏi
"thân người này đang thẳng đứng hay nằm ngang", dựa trên vị trí vai và hông.
Nhờ vậy có thể chặn báo ngã giả mà không cần huấn luyện lại model.

Thứ tự keypoint COCO (17 điểm):
    0 mũi, 1-2 mắt, 3-4 tai,
    5-6 vai, 7-8 khuỷu tay, 9-10 cổ tay,
    11-12 hông, 13-14 đầu gối, 15-16 cổ chân
"""

from __future__ import annotations

import math

import numpy as np

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

SHOULDERS = (LEFT_SHOULDER, RIGHT_SHOULDER)
HIPS = (LEFT_HIP, RIGHT_HIP)

DEFAULT_MIN_CONF = 0.30

# Vector "hướng lên" trong hệ toạ độ ảnh (trục y hướng xuống).
UP_VECTOR = np.array([0.0, -1.0])

MIN_TORSO_LENGTH_PX = 4.0


def _is_visible(keypoints: np.ndarray, index: int, min_conf: float) -> bool:
    """Keypoint được coi là nhìn rõ khi đủ tin cậy và có toạ độ hợp lệ."""
    x, y, conf = keypoints[index]
    return bool(conf >= min_conf and x > 0 and y > 0)


def _midpoint(
    keypoints: np.ndarray,
    indices: tuple[int, ...],
    min_conf: float,
) -> np.ndarray | None:
    """Trung điểm các keypoint nhìn rõ trong nhóm; None nếu không có điểm nào."""
    points = [keypoints[i][:2] for i in indices if _is_visible(keypoints, i, min_conf)]
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=np.float64), axis=0)


def has_torso(keypoints: np.ndarray, min_conf: float = DEFAULT_MIN_CONF) -> bool:
    """Có đủ vai và hông để xác định hướng thân người hay không.

    Khi người ở quá gần camera và chỉ có đầu hiện ra trong khung hình,
    hàm này trả về False — và đó chính là điều kiện để chặn báo ngã giả.
    """
    return (
        _midpoint(keypoints, SHOULDERS, min_conf) is not None
        and _midpoint(keypoints, HIPS, min_conf) is not None
    )


def torso_angle_deg(
    keypoints: np.ndarray,
    min_conf: float = DEFAULT_MIN_CONF,
) -> float | None:
    """Góc của trục thân người so với phương thẳng đứng.

    Trả về:
        ``0``   — thân hoàn toàn thẳng đứng (đang đứng/đi bộ)
        ``90``  — thân nằm ngang (ngã, nằm, bò)
        ``180`` — thân dốc ngược (treo đầu xuống)

    Trả về ``None`` khi không nhìn rõ đủ vai và hông.
    """
    shoulders = _midpoint(keypoints, SHOULDERS, min_conf)
    hips = _midpoint(keypoints, HIPS, min_conf)
    if shoulders is None or hips is None:
        return None

    torso = shoulders - hips
    length = float(np.linalg.norm(torso))
    if length < MIN_TORSO_LENGTH_PX:
        return None

    cosine = float(np.dot(torso / length, UP_VECTOR))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def direction_from_vertical_deg(
    keypoints: np.ndarray,
    min_conf: float = DEFAULT_MIN_CONF,
) -> float | None:
    """Độ lệch so với phương thẳng đứng, không phân biệt đầu hướng nào.

    Khác với :func:`torso_angle_deg`, hàm này trả về ``90`` cho cả hai chiều
    nằm ngang và ``0`` cho cả hai chiều thẳng đứng, dùng khi chỉ cần biết
    "thân có nằm ngang hay không".
    """
    angle = torso_angle_deg(keypoints, min_conf)
    if angle is None:
        return None
    return min(angle, 180.0 - angle)


def is_horizontal(
    keypoints: np.ndarray,
    min_angle_deg: float,
    min_conf: float = DEFAULT_MIN_CONF,
) -> bool:
    """Thân người có đang nằm ngang ít nhất ``min_angle_deg`` hay không."""
    direction = direction_from_vertical_deg(keypoints, min_conf)
    return direction is not None and direction >= min_angle_deg
