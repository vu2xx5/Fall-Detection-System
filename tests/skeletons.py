"""Bộ dựng khung xương tổng hợp 17 điểm để kiểm thử, không cần camera.

Toạ độ theo hệ ảnh: gốc ở góc trên trái, trục y hướng xuống.
"""

from __future__ import annotations

import numpy as np

KEYPOINT_COUNT = 17
HIGH_CONF = 0.9
LOW_CONF = 0.1

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4


def _blank() -> np.ndarray:
    return np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)


def _set(keypoints: np.ndarray, index: int, x: float, y: float, conf: float) -> None:
    keypoints[index] = (x, y, conf)


def standing(conf: float = HIGH_CONF) -> np.ndarray:
    """Người đứng thẳng, đủ 13 điểm như một khung xương thật."""
    kpts = _blank()
    _set(kpts, NOSE, 400, 160, conf)
    _set(kpts, LEFT_SHOULDER, 380, 200, conf)
    _set(kpts, RIGHT_SHOULDER, 420, 200, conf)
    _set(kpts, 7, 372, 260, conf)
    _set(kpts, 8, 428, 260, conf)
    _set(kpts, 9, 368, 320, conf)
    _set(kpts, 10, 432, 320, conf)
    _set(kpts, LEFT_HIP, 385, 300, conf)
    _set(kpts, RIGHT_HIP, 415, 300, conf)
    _set(kpts, 13, 388, 380, conf)
    _set(kpts, 14, 412, 380, conf)
    _set(kpts, 15, 388, 460, conf)
    _set(kpts, 16, 412, 460, conf)
    return kpts


def bent_over(conf: float = HIGH_CONF) -> np.ndarray:
    """Cúi gập người: trục thân nghiêng đúng 45 độ, chưa nằm ngang."""
    kpts = _blank()
    _set(kpts, LEFT_SHOULDER, 490, 195, conf)
    _set(kpts, RIGHT_SHOULDER, 510, 205, conf)
    _set(kpts, LEFT_HIP, 385, 295, conf)
    _set(kpts, RIGHT_HIP, 415, 305, conf)
    _set(kpts, NOSE, 545, 165, conf)
    return kpts


def lying(conf: float = HIGH_CONF) -> np.ndarray:
    """Người nằm ngang trên sàn, đầu quay về bên trái khung hình."""
    kpts = _blank()
    _set(kpts, NOSE, 160, 385, conf)
    _set(kpts, LEFT_SHOULDER, 190, 395, conf)
    _set(kpts, RIGHT_SHOULDER, 210, 405, conf)
    _set(kpts, 7, 155, 380, conf)
    _set(kpts, 8, 175, 390, conf)
    _set(kpts, 9, 130, 375, conf)
    _set(kpts, 10, 150, 385, conf)
    _set(kpts, LEFT_HIP, 340, 395, conf)
    _set(kpts, RIGHT_HIP, 360, 405, conf)
    _set(kpts, 13, 420, 400, conf)
    _set(kpts, 14, 440, 410, conf)
    _set(kpts, 15, 490, 405, conf)
    _set(kpts, 16, 510, 415, conf)
    return kpts


def upside_down(conf: float = HIGH_CONF) -> np.ndarray:
    """Treo ngược: vai ở phía dưới hông trong khung hình."""
    kpts = _blank()
    _set(kpts, LEFT_SHOULDER, 380, 400, conf)
    _set(kpts, RIGHT_SHOULDER, 420, 400, conf)
    _set(kpts, LEFT_HIP, 385, 300, conf)
    _set(kpts, RIGHT_HIP, 415, 300, conf)
    return kpts


def head_only(conf: float = HIGH_CONF) -> np.ndarray:
    """Chỉ nhìn thấy đầu — trường hợp người ở quá gần camera."""
    kpts = _blank()
    _set(kpts, NOSE, 400, 160, conf)
    _set(kpts, LEFT_EYE, 392, 152, conf)
    _set(kpts, RIGHT_EYE, 408, 152, conf)
    _set(kpts, LEFT_EAR, 380, 158, conf)
    _set(kpts, RIGHT_EAR, 420, 158, conf)
    return kpts


def side_view_lying(conf: float = HIGH_CONF) -> np.ndarray:
    """Nhìn nghiêng khi nằm: chỉ một vai và một hông hiện rõ."""
    kpts = _blank()
    _set(kpts, LEFT_SHOULDER, 200, 400, conf)
    _set(kpts, LEFT_HIP, 350, 400, conf)
    _set(kpts, NOSE, 170, 385, conf)
    return kpts


def head_and_arms(conf: float = HIGH_CONF) -> np.ndarray:
    """Chỉ thấy đầu và hai tay: đủ điểm để vào bộ đệm nhưng thiếu vai và hông."""
    kpts = _blank()
    _set(kpts, NOSE, 400, 200, conf)
    _set(kpts, LEFT_EYE, 392, 192, conf)
    _set(kpts, RIGHT_EYE, 408, 192, conf)
    _set(kpts, LEFT_EAR, 380, 198, conf)
    _set(kpts, RIGHT_EAR, 420, 198, conf)
    _set(kpts, 7, 350, 260, conf)
    _set(kpts, 8, 450, 260, conf)
    _set(kpts, 9, 330, 320, conf)
    _set(kpts, 10, 470, 320, conf)
    return kpts


def sitting_floor(conf: float = HIGH_CONF) -> np.ndarray:
    """Người ngồi bệt trên sàn: thân thẳng đứng nhưng chân co sát hông."""
    kpts = _blank()
    _set(kpts, NOSE, 400, 310, conf)
    _set(kpts, LEFT_SHOULDER, 380, 350, conf)
    _set(kpts, RIGHT_SHOULDER, 420, 350, conf)
    _set(kpts, LEFT_HIP, 385, 450, conf)
    _set(kpts, RIGHT_HIP, 415, 450, conf)
    _set(kpts, 13, 388, 430, conf)
    _set(kpts, 14, 412, 430, conf)
    _set(kpts, 15, 388, 460, conf)
    _set(kpts, 16, 412, 460, conf)
    return kpts


def sitting_chair(conf: float = HIGH_CONF) -> np.ndarray:
    """Người ngồi ghế: đùi vuông góc với thân, cẳng chân thõng xuống."""
    kpts = _blank()
    _set(kpts, NOSE, 400, 210, conf)
    _set(kpts, LEFT_SHOULDER, 380, 250, conf)
    _set(kpts, RIGHT_SHOULDER, 420, 250, conf)
    _set(kpts, LEFT_HIP, 385, 350, conf)
    _set(kpts, RIGHT_HIP, 415, 350, conf)
    _set(kpts, 13, 440, 355, conf)
    _set(kpts, 14, 460, 355, conf)
    _set(kpts, 15, 440, 430, conf)
    _set(kpts, 16, 460, 430, conf)
    return kpts
