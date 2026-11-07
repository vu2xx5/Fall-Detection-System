"""Trạng thái theo dõi của một người, gắn với một ``track_id`` của ByteTrack."""

from __future__ import annotations

from collections import deque

import numpy as np

from fall_verifier import FallVerifier, Verdict, VerdictReason

DEFAULT_SEQ_LEN = 30


class PersonTrackState:
    """Bộ đệm chuỗi keypoint, xác suất ngã và bộ xác minh của một người.

    Mỗi người có bộ đệm riêng nên trạng thái của người này không ảnh hưởng
    tới người khác, kể cả khi họ đi qua nhau trong khung hình.
    """

    def __init__(self, seq_len: int = DEFAULT_SEQ_LEN) -> None:
        self.buffer: deque[np.ndarray] = deque(maxlen=seq_len)
        self.seq_len = seq_len
        self.fall_prob = 0.0
        self.verdict = Verdict(False, VerdictReason.MODEL_NORMAL)
        self.last_seen_frame = 0
        self._verifier = FallVerifier()

    def configure_verifier(self, verifier: FallVerifier) -> None:
        """Gắn bộ xác minh đã cấu hình từ ``Settings``."""
        self._verifier = verifier

    def record_frame(self, feature: np.ndarray) -> None:
        """Thêm vector đặc trưng 52 chiều của một frame vào bộ đệm."""
        self.buffer.append(feature)

    @property
    def is_ready(self) -> bool:
        """Đã tích luỹ đủ ``seq_len`` frame để chạy dự đoán chưa."""
        return len(self.buffer) == self.seq_len

    @property
    def progress(self) -> tuple[int, int]:
        """Cặp ``(số frame đã có, số frame cần)`` để hiển thị tiến độ."""
        return len(self.buffer), self.seq_len

    def sequence(self) -> np.ndarray:
        """Toàn bộ bộ đệm dưới dạng mảng ``(seq_len, 52)``."""
        return np.array(self.buffer, dtype=np.float32)

    @property
    def is_fallen(self) -> bool:
        """Chỉ True khi bộ xác minh đã xác nhận ngã."""
        return self.verdict.confirmed

    @property
    def reason(self) -> str:
        """Lý do của kết luận hiện tại, dùng cho nhãn hiển thị."""
        return self.verdict.reason.value

    @property
    def seconds_fallen(self) -> float | None:
        """Thời gian đã ngã tính bằng giây (None nếu chưa ngã)."""
        return self._verifier.seconds_fallen

    def tick(
        self,
        now: float,
        torso_angle_deg: float | None,
        fall_prob: float | None = None,
        is_sitting: bool = False,
    ) -> Verdict:
        """Cập nhật một mẫu quan sát tại thời điểm ``now`` (giây).

        ``fall_prob`` chỉ truyền vào khi vừa có dự đoán mới; các frame còn lại
        dùng lại xác suất gần nhất để đồng hồ thời gian vẫn chạy liên tục.
        """
        if fall_prob is not None:
            self.fall_prob = float(fall_prob)

        self.verdict = self._verifier.update(
            now,
            torso_angle_deg,
            self.fall_prob,
            is_sitting=is_sitting,
        )
        return self.verdict
