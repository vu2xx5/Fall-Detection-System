"""Kiểm thử tích hợp: chạy đúng luồng ``main._step_people`` với model giả lập.

Model giả lập đọc trực tiếp ``aspect_ratio`` (phần tử thứ 52 của vector đặc
trưng) để quyết định xác suất, nên nó tái hiện được đúng kiểu sai mà mạng
Bi-LSTM hay mắc: thấy khung xương "dẹt" là cho rằng người đã ngã.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import main
import skeletons
from config import Settings
from fall_verifier import Verdict
from track_state import PersonTrackState

FPS = 30
FRAME_STEP_SEC = 1.0 / FPS
FRAME_STRIDE = 2
SEQ_LEN = 30

ASPECT_RATIO_INDEX = 51
ASPECT_RATIO_TRIGGER = 2.0

PROB_HIGH = 0.95
PROB_LOW = 0.02

FALL_START_SEC = 4.0
SIMULATION_SEC = 9.0


def _logit(prob: float) -> float:
    return math.log(prob / (1.0 - prob))


class AspectRatioModel:
    """Model giả: coi khung xương dẹt (aspect_ratio lớn) là ngã."""

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        aspect = batch[:, -1, ASPECT_RATIO_INDEX]
        high = torch.full_like(aspect, _logit(PROB_HIGH))
        low = torch.full_like(aspect, _logit(PROB_LOW))
        return torch.where(aspect > ASPECT_RATIO_TRIGGER, high, low)


def make_settings() -> Settings:
    return Settings(
        video_source=0,
        lstm_weights="unused.pth",
        yolo_weights="unused.pt",
        telegram_bot_token="",
        telegram_chat_id="",
        alert_cooldown_sec=60,
        max_inactive_frames=150,
        min_valid_keypoints=6,
        min_conf_keypoint=0.30,
        fall_prob_threshold=0.75,
        fall_prob_reset=0.40,
        upright_torso_angle_deg=35.0,
        fall_torso_angle_deg=55.0,
        fall_transition_sec=1.5,
        fall_hold_sec=1.0,
    )


def make_states(count: int, settings: Settings) -> list[PersonTrackState]:
    states = []
    for _ in range(count):
        state = PersonTrackState(seq_len=SEQ_LEN)
        state.configure_verifier(settings.build_verifier())
        states.append(state)
    return states


def simulate(
    builders,
    settings: Settings,
    duration_sec: float = SIMULATION_SEC,
) -> tuple[list[list[Verdict]], list[PersonTrackState]]:
    """Chạy mô phỏng, trả về (lịch sử verdict theo frame, trạng thái cuối)."""
    states = make_states(len(builders), settings)
    model = AspectRatioModel()
    device = torch.device("cpu")
    verdict_history: list[list[Verdict]] = []

    total_frames = int(duration_sec * FPS)
    for frame_idx in range(1, total_frames + 1):
        now = round(frame_idx * FRAME_STEP_SEC, 6)
        keypoints = np.stack([builder(now) for builder in builders])
        main._step_people(
            states,
            keypoints,
            frame_idx,
            FRAME_STRIDE,
            settings,
            model,
            device,
            now,
        )
        verdict_history.append([s.verdict for s in states])

    return verdict_history, states


def first_fallen_time(verdict_history: list[list[Verdict]], person: int) -> float | None:
    """Thời điểm (giây) người thứ ``person`` lần đầu bị kết luận ngã."""
    for offset, verdicts in enumerate(verdict_history):
        if verdicts[person].confirmed:
            return (offset + 1) * FRAME_STEP_SEC
    return None


class TestWalkingIsNeverFall:
    def test_walking_person_never_confirmed(self) -> None:
        settings = make_settings()
        verdict_history, states = simulate([lambda _now: skeletons.standing()], settings)

        assert first_fallen_time(verdict_history, 0) is None
        assert states[0].is_ready is True

    def test_model_shouting_fall_cannot_override_posture(self) -> None:
        """Model giả luôn nói rất chắc là ngã, nhưng người vẫn đang đứng."""
        settings = make_settings()
        builders = [lambda _now: skeletons.standing()]
        verdict_history, states = simulate(builders, settings)

        assert states[0].fall_prob < settings.fall_prob_threshold
        assert first_fallen_time(verdict_history, 0) is None


class TestRealFallIsDetected:
    def test_fast_fall_confirmed_after_hold(self) -> None:
        settings = make_settings()

        def person(now: float) -> np.ndarray:
            return skeletons.lying() if now >= FALL_START_SEC else skeletons.standing()

        verdict_history, states = simulate([person], settings)
        fallen_at = first_fallen_time(verdict_history, 0)

        assert fallen_at is not None
        assert fallen_at >= FALL_START_SEC + settings.fall_hold_sec

    def test_bystander_stays_normal(self) -> None:
        """Một người ngã không được làm người bên cạnh bị báo ngã theo."""
        settings = make_settings()

        def faller(now: float) -> np.ndarray:
            return skeletons.lying() if now >= FALL_START_SEC else skeletons.standing()

        def bystander(_now: float) -> np.ndarray:
            return skeletons.standing()

        verdict_history, states = simulate([faller, bystander], settings)

        assert first_fallen_time(verdict_history, 0) is not None
        assert first_fallen_time(verdict_history, 1) is None


class TestFaceOnlyIsNeverFall:
    def test_head_and_arms_never_confirmed(self) -> None:
        """Lỗi người dùng báo: chỉ thấy mặt/gần camera mà hệ thống báo ngã."""
        settings = make_settings()

        def person(now: float) -> np.ndarray:
            return skeletons.head_and_arms() if now >= FALL_START_SEC else skeletons.standing()

        verdict_history, states = simulate([person], settings)

        assert first_fallen_time(verdict_history, 0) is None
        assert states[0].is_ready is True
        assert states[0].reason == "no_torso"


class TestSlowLieDownIsNeverFall:
    def test_slow_lie_down_never_confirmed(self) -> None:
        settings = make_settings()

        def person(now: float) -> np.ndarray:
            if now < FALL_START_SEC:
                return skeletons.standing()
            if now < FALL_START_SEC + 2.5:
                return skeletons.bent_over()
            return skeletons.lying()

        verdict_history, states = simulate([person], settings)

        assert first_fallen_time(verdict_history, 0) is None
        assert states[0].reason == "slow_lie_down"


class TestThresholdIsConfigurable:
    def test_stricter_probability_threshold_blocks_alert(self) -> None:
        base = make_settings()
        strict = Settings(
            video_source=base.video_source,
            lstm_weights=base.lstm_weights,
            yolo_weights=base.yolo_weights,
            telegram_bot_token="",
            telegram_chat_id="",
            alert_cooldown_sec=base.alert_cooldown_sec,
            max_inactive_frames=base.max_inactive_frames,
            min_valid_keypoints=base.min_valid_keypoints,
            min_conf_keypoint=base.min_conf_keypoint,
            fall_prob_threshold=0.99,
            fall_prob_reset=0.90,
            upright_torso_angle_deg=base.upright_torso_angle_deg,
            fall_torso_angle_deg=base.fall_torso_angle_deg,
            fall_transition_sec=base.fall_transition_sec,
            fall_hold_sec=base.fall_hold_sec,
        )

        def person(now: float) -> np.ndarray:
            return skeletons.lying() if now >= FALL_START_SEC else skeletons.standing()

        verdict_history, states = simulate([person], strict)

        assert first_fallen_time(verdict_history, 0) is None


class TestBufferWarmup:
    def test_not_ready_before_enough_frames(self) -> None:
        settings = make_settings()
        state = make_states(1, settings)[0]

        assert state.is_ready is False
        assert state.progress == (0, SEQ_LEN)

        for _ in range(SEQ_LEN):
            state.record_frame(np.zeros(52, dtype=np.float32))

        assert state.is_ready is True
        assert state.sequence().shape == (SEQ_LEN, 52)
