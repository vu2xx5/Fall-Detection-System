"""Bộ xác minh ngã: kết hợp mô hình Bi-LSTM với hình học và động học tư thế.

Mô hình chỉ trả về xác suất; nó không biết thân người có thật sự nằm ngang,
cũng không biết tư thế thay đổi nhanh hay chậm. Lớp này bổ sung hai điều đó:

1. **Hình học** — phải nhìn rõ vai và hông, và trục thân phải nằm ngang.
   Chỉ thấy mặt thì không bao giờ kết luận ngã.
2. **Động học** — tư thế phải chuyển từ thẳng đứng sang nằm ngang trong thời
   gian ngắn. Người từ từ nằm xuống (``> FALL_TRANSITION_SEC``) bị loại.

Cuối cùng, cả tư thế nằm ngang và xác suất cao đều phải được giữ liên tục ít
nhất ``FALL_HOLD_SEC`` giây mới xác nhận. Đây là điểm khác biệt cốt lõi so với
cách đếm theo frame trước kia: 3 frame chỉ là ~0,1 giây, còn 1 giây là 1 giây
thật, nên một gai nhiễu đơn lẻ không thể kích hoạt cảnh báo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VerdictReason(str, Enum):
    """Lý do của một kết luận, dùng để hiển thị và gỡ lỗi."""

    MODEL_NORMAL = "model_normal"
    NO_TORSO = "no_torso"
    UPRIGHT = "upright"
    SITTING = "sitting"
    AMBIGUOUS = "ambiguous"
    NO_TRANSITION = "no_upright_observed"
    SLOW_LIE_DOWN = "slow_lie_down"
    CONFIRMING = "confirming"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class Verdict:
    """Kết quả xác minh cho một người tại một thời điểm."""

    confirmed: bool
    reason: VerdictReason


class FallVerifier:
    """Máy trạng thái xác nhận ngã cho **một** người, dựa trên thời gian thực.

    Mọi mốc thời gian tính bằng giây (``time.monotonic()``), không phải frame,
    nên hoạt động đúng dù camera 15 hay 60 FPS, và dù có frame bị bỏ qua.
    """

    def __init__(
        self,
        prob_threshold: float = 0.75,
        prob_reset: float = 0.40,
        upright_angle_deg: float = 35.0,
        horizontal_angle_deg: float = 55.0,
        transition_sec: float = 1.5,
        hold_sec: float = 1.0,
        recovery_sec: float = 2.0,
    ) -> None:
        if prob_reset > prob_threshold:
            raise ValueError("prob_reset phải nhỏ hơn hoặc bằng prob_threshold")
        if upright_angle_deg >= horizontal_angle_deg:
            raise ValueError("upright_angle_deg phải nhỏ hơn horizontal_angle_deg")

        self._prob_threshold = prob_threshold
        self._prob_reset = prob_reset
        self._upright_angle = upright_angle_deg
        self._horizontal_angle = horizontal_angle_deg
        self._transition_sec = transition_sec
        self._hold_sec = hold_sec
        self._recovery_sec = recovery_sec

        self._high_prob_since: float | None = None
        self._horizontal_since: float | None = None
        self._last_upright_time: float | None = None
        self._last_sitting_time: float | None = None
        self._last_update_time: float | None = None
        self._prev_angle: float | None = None

        # Trạng thái chốt ngã (Latched fallen state)
        self._is_fallen: bool = False
        self._fallen_since: float | None = None
        self._standing_since: float | None = None

    def reset(self) -> None:
        """Xoá toàn bộ trạng thái thời gian (dùng khi mất dấu người)."""
        self._high_prob_since = None
        self._horizontal_since = None
        self._last_upright_time = None
        self._last_sitting_time = None
        self._last_update_time = None
        self._prev_angle = None
        self._is_fallen = False
        self._fallen_since = None
        self._standing_since = None

    def update(
        self,
        now: float,
        torso_angle_deg: float | None,
        fall_prob: float,
        is_sitting: bool = False,
    ) -> Verdict:
        """Nạp một mẫu quan sát và trả về kết luận.

        Tham số:
            now: thời điểm hiện tại, tính bằng giây.
            torso_angle_deg: góc thân so với phương thẳng đứng, ``None`` nếu
                không nhìn rõ vai và hông.
            fall_prob: xác suất ngã do Bi-LSTM dự đoán, trong ``[0, 1]``.
            is_sitting: đối tượng có đang ở tư thế ngồi hay không.
        """
        self._update_prob_clock(now, fall_prob)

        # 1. NẾU ĐÃ Ở TRẠNG THÁI CHỐT NGÃ:
        # Bắt buộc duy trì cảnh báo ĐỎ chừng nào đối tượng chưa đứng dậy vững vàng.
        if self._is_fallen:
            self._last_update_time = now
            # Kiểm tra xem đối tượng có đang đứng dậy thật sự hay không
            if (
                torso_angle_deg is not None
                and torso_angle_deg <= self._upright_angle
                and not is_sitting
            ):
                if self._standing_since is None:
                    self._standing_since = now
                elif now - self._standing_since >= self._recovery_sec:
                    # Đã đứng dậy vững vàng đủ recovery_sec giây -> HỒI PHỤC!
                    self._is_fallen = False
                    self._fallen_since = None
                    self._standing_since = None
                    self._last_upright_time = now
                    self._horizontal_since = None
                    self._high_prob_since = None
                    return Verdict(False, VerdictReason.UPRIGHT)
            else:
                # Vẫn đang nằm, quẫy đạp hoặc ngồi bệt dưới sàn -> reset đồng hồ đứng
                self._standing_since = None

            return Verdict(True, VerdictReason.CONFIRMED)

        # 2. CHƯA NGÃ: PHÂN TÍCH TƯ THẾ VÀ CÚ CHUYỂN ĐỘNG
        if torso_angle_deg is None:
            self._last_update_time = now
            return Verdict(False, VerdictReason.NO_TORSO)

        if torso_angle_deg <= self._upright_angle:
            self._horizontal_since = None
            self._prev_angle = torso_angle_deg
            self._last_update_time = now

            if is_sitting:
                # Đang ngồi lưng thẳng: ghi nhận thời gian ngồi, không ghi nhận là đứng
                self._last_sitting_time = now
                return Verdict(False, VerdictReason.SITTING)

            # Đang đứng thẳng (standing)
            self._last_upright_time = now
            return Verdict(False, VerdictReason.UPRIGHT)

        if torso_angle_deg < self._horizontal_angle:
            # Vùng chuyển tiếp (cúi người, khom lưng): giữ nguyên các mốc thời gian
            self._prev_angle = torso_angle_deg
            self._last_update_time = now
            return Verdict(False, VerdictReason.AMBIGUOUS)

        # Thân người đang nằm ngang (torso_angle_deg >= self._horizontal_angle)
        if self._horizontal_since is None:
            if self._last_upright_time is None:
                # Chưa từng thấy người này ở tư thế đứng (người đã nằm sẵn từ đầu)
                self._prev_angle = torso_angle_deg
                self._last_update_time = now
                return Verdict(False, VerdictReason.NO_TRANSITION)

            # Chặn báo ngã nếu vừa ở tư thế ngồi gần đây (ngồi xuống rồi ngả lưng nằm)
            if self._last_sitting_time is not None and (now - self._last_sitting_time <= 3.0):
                self._prev_angle = torso_angle_deg
                self._last_update_time = now
                return Verdict(False, VerdictReason.SLOW_LIE_DOWN)

            # Chặn nếu thời gian chuyển từ đứng sang nằm quá chậm (> transition_sec)
            if now - self._last_upright_time > self._transition_sec:
                self._prev_angle = torso_angle_deg
                self._last_update_time = now
                return Verdict(False, VerdictReason.SLOW_LIE_DOWN)

            self._horizontal_since = now

        horizontal_for = now - self._horizontal_since
        prob_for = 0.0 if self._high_prob_since is None else now - self._high_prob_since

        self._prev_angle = torso_angle_deg
        self._last_update_time = now

        if horizontal_for >= self._hold_sec and prob_for >= self._hold_sec:
            # Xác nhận cú ngã thành công và kích hoạt chốt trạng thái ngã
            self._is_fallen = True
            self._fallen_since = now
            return Verdict(True, VerdictReason.CONFIRMED)

        return Verdict(False, VerdictReason.CONFIRMING)

    def _update_prob_clock(self, now: float, fall_prob: float) -> None:
        """Đồng hồ xác suất có trễ (hysteresis) để chống dao động quanh ngưỡng."""
        if fall_prob >= self._prob_threshold:
            if self._high_prob_since is None:
                self._high_prob_since = now
        elif fall_prob < self._prob_reset:
            self._high_prob_since = None

    @property
    def seconds_horizontal(self) -> float | None:
        """Đã nằm ngang bao lâu tại lần cập nhật gần nhất (None nếu chưa)."""
        if self._horizontal_since is None or self._last_update_time is None:
            return None
        return self._last_update_time - self._horizontal_since

    @property
    def is_fallen(self) -> bool:
        """Trạng thái chốt ngã hiện tại."""
        return self._is_fallen

    @property
    def seconds_fallen(self) -> float | None:
        """Đã ngã được bao nhiêu giây (None nếu chưa ngã)."""
        if not self._is_fallen or self._fallen_since is None or self._last_update_time is None:
            return None
        return max(0.0, self._last_update_time - self._fallen_since)
