"""Kiểm thử hình học tư thế trên khung xương tổng hợp."""

from __future__ import annotations

import pytest

import posture
import skeletons


class TestStanding:
    def test_angle_near_zero(self) -> None:
        angle = posture.direction_from_vertical_deg(skeletons.standing())
        assert angle is not None
        assert angle == pytest.approx(0.0, abs=1.0)

    def test_has_torso(self) -> None:
        assert posture.has_torso(skeletons.standing()) is True

    def test_not_horizontal(self) -> None:
        assert posture.is_horizontal(skeletons.standing(), min_angle_deg=55.0) is False


class TestLying:
    def test_angle_near_ninety(self) -> None:
        angle = posture.direction_from_vertical_deg(skeletons.lying())
        assert angle is not None
        assert angle == pytest.approx(90.0, abs=1.0)

    def test_is_horizontal(self) -> None:
        assert posture.is_horizontal(skeletons.lying(), min_angle_deg=55.0) is True

    def test_side_view_still_detected(self) -> None:
        """Chỉ thấy một vai và một hông vẫn phải xác định được tư thế nằm."""
        assert posture.has_torso(skeletons.side_view_lying()) is True
        angle = posture.direction_from_vertical_deg(skeletons.side_view_lying())
        assert angle is not None
        assert angle == pytest.approx(90.0, abs=1.0)


class TestBentOver:
    def test_bent_is_ambiguous(self) -> None:
        """Cúi gập người không được coi là nằm ngang."""
        angle = posture.direction_from_vertical_deg(skeletons.bent_over())
        assert angle is not None
        assert 35.0 < angle < 60.0
        assert posture.is_horizontal(skeletons.bent_over(), min_angle_deg=60.0) is False


class TestUpsideDown:
    def test_angle_near_180(self) -> None:
        angle = posture.torso_angle_deg(skeletons.upside_down())
        assert angle is not None
        assert angle == pytest.approx(180.0, abs=1.0)

    def test_direction_collapses_direction(self) -> None:
        """Hướng không phân biệt: treo ngược được coi như thẳng đứng."""
        assert posture.direction_from_vertical_deg(skeletons.upside_down()) == pytest.approx(0.0, abs=1.0)


class TestHeadOnly:
    def test_has_torso_false(self) -> None:
        """Chỉ thấy mặt thì không thể xác định tư thế thân."""
        assert posture.has_torso(skeletons.head_only()) is False

    def test_angle_is_none(self) -> None:
        assert posture.torso_angle_deg(skeletons.head_only()) is None
        assert posture.direction_from_vertical_deg(skeletons.head_only()) is None

    def test_never_horizontal(self) -> None:
        """Đây là điều kiện chặn báo ngã giả khi người ở gần camera."""
        assert posture.is_horizontal(skeletons.head_only(), min_angle_deg=10.0) is False


class TestLowConfidence:
    def test_low_conf_keypoints_ignored(self) -> None:
        kpts = skeletons.standing(conf=skeletons.LOW_CONF)
        assert posture.has_torso(kpts) is False
        assert posture.torso_angle_deg(kpts) is None

    def test_threshold_is_configurable(self) -> None:
        kpts = skeletons.standing(conf=0.2)
        assert posture.torso_angle_deg(kpts, min_conf=0.3) is None
        assert posture.torso_angle_deg(kpts, min_conf=0.15) is not None
