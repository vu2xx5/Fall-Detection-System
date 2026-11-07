"""Kiểm thử bộ xác minh ngã: hình học, động học và thời gian giữ."""

from __future__ import annotations

import pytest

from fall_verifier import FallVerifier, Verdict, VerdictReason

UPRIGHT_ANGLE = 5.0
BENT_ANGLE = 45.0
HORIZONTAL_ANGLE = 88.0

PROB_LOW = 0.10
PROB_MID = 0.50
PROB_HIGH = 0.92

FRAME_DT_30FPS = 1.0 / 30.0


def timeline(
    start: float,
    end: float,
    step: float,
    torso_angle: float | None,
    prob: float,
) -> list[tuple[float, float | None, float]]:
    """Sinh chuỗi mẫu quan sát đều nhau giữa ``start`` và ``end`` (giây)."""
    samples: list[tuple[float, float | None, float]] = []
    moment = start
    while moment <= end + 1e-9:
        samples.append((round(moment, 6), torso_angle, prob))
        moment += step
    return samples


def run(
    verifier: FallVerifier,
    samples: list[tuple[float, float | None, float]],
) -> list[Verdict]:
    return [verifier.update(moment, angle, prob) for moment, angle, prob in samples]


def any_confirmed(verdicts: list[Verdict]) -> bool:
    return any(verdict.confirmed for verdict in verdicts)


class TestFastFall:
    """Ngã thật: đang đứng rồi đổ xuống nhanh, model đồng ý."""

    def test_confirms_after_hold(self) -> None:
        samples = timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples += timeline(1.1, 3.5, 0.1, HORIZONTAL_ANGLE, PROB_HIGH)

        verdicts = run(FallVerifier(), samples)

        assert verdicts[-1].confirmed is True
        assert verdicts[-1].reason is VerdictReason.CONFIRMED

    def test_not_confirmed_immediately_after_collapse(self) -> None:
        samples = timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples.append((1.1, HORIZONTAL_ANGLE, PROB_HIGH))

        verdicts = run(FallVerifier(), samples)

        assert verdicts[-1].confirmed is False
        assert verdicts[-1].reason is VerdictReason.CONFIRMING

    def test_three_frames_at_30fps_are_not_enough(self) -> None:
        """Chống hồi quy: cách đếm streak cũ coi 3 frame (~0,1 giây) là đủ."""
        samples = timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples += [
            (1.0 + FRAME_DT_30FPS, HORIZONTAL_ANGLE, PROB_HIGH),
            (1.0 + 2 * FRAME_DT_30FPS, HORIZONTAL_ANGLE, PROB_HIGH),
            (1.0 + 3 * FRAME_DT_30FPS, HORIZONTAL_ANGLE, PROB_HIGH),
        ]

        verdicts = run(FallVerifier(), samples)

        assert any_confirmed(verdicts) is False


class TestWalkingFalsePositive:
    """Lỗi người dùng báo: đang đi bộ mà hệ thống báo ngã."""

    def test_walking_with_high_prob_never_confirms(self) -> None:
        samples = timeline(0.0, 12.0, 0.1, UPRIGHT_ANGLE, PROB_HIGH)

        verdicts = run(FallVerifier(), samples)

        assert any_confirmed(verdicts) is False
        assert all(verdict.reason is VerdictReason.UPRIGHT for verdict in verdicts)

    def test_bent_over_is_not_horizontal(self) -> None:
        samples = timeline(0.0, 12.0, 0.1, BENT_ANGLE, PROB_HIGH)

        verdicts = run(FallVerifier(), samples)

        assert any_confirmed(verdicts) is False
        assert verdicts[-1].reason is VerdictReason.AMBIGUOUS


class TestSlowLieDown:
    """Người từ từ nằm xuống không được coi là ngã."""

    def test_slow_transition_rejected(self) -> None:
        samples = timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples += timeline(1.5, 3.5, 0.1, BENT_ANGLE, PROB_MID)
        samples += timeline(4.0, 9.0, 0.1, HORIZONTAL_ANGLE, PROB_HIGH)

        verdicts = run(FallVerifier(), samples)

        assert any_confirmed(verdicts) is False
        assert verdicts[-1].reason is VerdictReason.SLOW_LIE_DOWN

    def test_fast_collapse_after_standing_still_confirms(self) -> None:
        """Sau khi đứng dậy, một cú ngã nhanh vẫn phải được phát hiện."""
        samples = timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples += timeline(1.1, 2.0, 0.1, HORIZONTAL_ANGLE, PROB_HIGH)
        samples += timeline(2.1, 5.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples += timeline(5.1, 8.0, 0.1, HORIZONTAL_ANGLE, PROB_HIGH)

        verdicts = run(FallVerifier(), samples)

        assert verdicts[-1].confirmed is True


class TestHeadOnlyFalsePositive:
    """Lỗi người dùng báo: chỉ thấy mặt mà báo ngã."""

    def test_missing_torso_never_confirms(self) -> None:
        samples = timeline(0.0, 12.0, 0.1, None, PROB_HIGH)

        verdicts = run(FallVerifier(), samples)

        assert any_confirmed(verdicts) is False
        assert all(verdict.reason is VerdictReason.NO_TORSO for verdict in verdicts)


class TestNoUprightObserved:
    def test_already_lying_when_first_seen(self) -> None:
        """Không phân biệt được người vốn đã nằm sẵn, nên không báo động."""
        samples = timeline(0.0, 10.0, 0.1, HORIZONTAL_ANGLE, PROB_HIGH)

        verdicts = run(FallVerifier(), samples)

        assert any_confirmed(verdicts) is False
        assert verdicts[-1].reason is VerdictReason.NO_TRANSITION


class TestProbabilityGate:
    def test_below_threshold_blocks_confirmation(self) -> None:
        samples = timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples += timeline(1.1, 6.0, 0.1, HORIZONTAL_ANGLE, PROB_MID)

        verdicts = run(FallVerifier(), samples)

        assert any_confirmed(verdicts) is False
        assert verdicts[-1].reason is VerdictReason.CONFIRMING

    def test_single_probability_spike_does_not_confirm(self) -> None:
        samples = timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples.append((1.1, HORIZONTAL_ANGLE, PROB_HIGH))
        samples += timeline(1.2, 5.0, 0.1, HORIZONTAL_ANGLE, PROB_LOW)

        verdicts = run(FallVerifier(), samples)

        assert any_confirmed(verdicts) is False

    def test_hysteresis_keeps_clock_through_dip(self) -> None:
        """Xác suất dao động nhẹ quanh ngưỡng không được làm đứt đồng hồ."""
        samples = timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW)
        samples += timeline(1.1, 1.5, 0.1, HORIZONTAL_ANGLE, PROB_HIGH)
        samples += timeline(1.6, 1.8, 0.1, HORIZONTAL_ANGLE, PROB_MID)
        samples += timeline(1.9, 3.0, 0.1, HORIZONTAL_ANGLE, PROB_HIGH)

        verdicts = run(FallVerifier(), samples)

        assert verdicts[-1].confirmed is True


class TestReset:
    def test_reset_clears_timers(self) -> None:
        verifier = FallVerifier()
        run(verifier, timeline(0.0, 1.0, 0.1, UPRIGHT_ANGLE, PROB_LOW))
        run(verifier, timeline(1.1, 1.5, 0.1, HORIZONTAL_ANGLE, PROB_HIGH))

        verifier.reset()

        verdict = verifier.update(1.6, HORIZONTAL_ANGLE, PROB_HIGH)
        assert verdict.confirmed is False
        assert verdict.reason is VerdictReason.NO_TRANSITION


class TestConstructorValidation:
    def test_rejects_reset_above_threshold(self) -> None:
        with pytest.raises(ValueError):
            FallVerifier(prob_threshold=0.5, prob_reset=0.8)

    def test_rejects_inverted_angles(self) -> None:
        with pytest.raises(ValueError):
            FallVerifier(upright_angle_deg=70.0, horizontal_angle_deg=55.0)
