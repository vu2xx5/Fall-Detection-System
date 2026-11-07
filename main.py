"""Điểm khởi chạy: phát hiện té ngã đa đối tượng theo thời gian thực.

Kiến trúc: YOLOv8-Pose + ByteTrack lấy khung xương và ID từng người,
Bi-LSTM + Attention dự đoán trên chuỗi 30 frame, và :class:`FallVerifier`
(hình học + thời gian) chỉ xác nhận ngã khi thân người thật sự nằm ngang
sau một cú chuyển tư thế nhanh.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import posture
from posture import has_torso
from config import Settings, load_settings
from model import (
    FEATURE_DIM,
    FRAME_STRIDE,
    SEQ_LEN,
    FallDetectionLSTM,
    normalize_single_frame_keypoints,
)
from notifier import FallAlertNotifier
from renderer import draw_hud, draw_person
from track_state import PersonTrackState

WINDOW_TITLE = "Multi-Person Fall Detection System (Bi-LSTM + YOLOv8-Pose)"
YOLO_CONFIDENCE = 0.25
STALE_CHECK_INTERVAL = 60


def _load_lstm(
    settings: Settings,
    device: torch.device,
) -> tuple[FallDetectionLSTM, int, int]:
    """Nạp mô hình Bi-LSTM từ file trọng số và đọc siêu tham số đã lưu kèm."""
    checkpoint = torch.load(settings.lstm_weights, map_location=device)
    seq_len = int(checkpoint.get("seq_len", SEQ_LEN))
    feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
    frame_stride = int(checkpoint.get("frame_stride", FRAME_STRIDE))

    model = FallDetectionLSTM(input_dim=feature_dim, seq_len=seq_len).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Đã nạp {settings.lstm_weights} (Seq_Len={seq_len}, Stride={frame_stride})")
    return model, seq_len, frame_stride


def _predict_batch(
    model: FallDetectionLSTM,
    sequences: list[np.ndarray],
    device: torch.device,
) -> list[float]:
    """Chạy Bi-LSTM cho nhiều người trong một lần forward, trả về xác suất ngã."""
    if not sequences:
        return []

    batch = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
    return np.atleast_1d(probs).astype(float).tolist()


def _step_people(
    states: list[PersonTrackState],
    keypoints: np.ndarray,
    frame_idx: int,
    frame_stride: int,
    settings: Settings,
    model: FallDetectionLSTM,
    device: torch.device,
    now: float,
) -> None:
    """Cập nhật bộ đệm, chạy dự đoán và cập nhật kết luận cho từng người."""
    angles = [
        posture.direction_from_vertical_deg(person_keypoints, settings.min_conf_keypoint)
        for person_keypoints in keypoints
    ]
    torso_visible = [
        has_torso(person_keypoints, settings.min_conf_keypoint)
        for person_keypoints in keypoints
    ]
    sitting_flags = [
        posture.is_sitting(person_keypoints, settings.min_conf_keypoint)
        for person_keypoints in keypoints
    ]

    for index, state in enumerate(states):
        if frame_idx % frame_stride != 0:
            continue
        feature, valid_kpts, _ = normalize_single_frame_keypoints(keypoints[index])
        if valid_kpts >= settings.min_valid_keypoints:
            state.record_frame(feature)

    # Chỉ chạy LSTM cho những người có torso nhìn rõ — chặn false positive
    # khi chỉ thấy mặt/đầu gần camera.
    ready_indices = [
        i for i, state in enumerate(states)
        if state.is_ready and torso_visible[i]
    ]
    probabilities = _predict_batch(
        model,
        [states[i].sequence() for i in ready_indices],
        device,
    )
    prob_by_index = dict(zip(ready_indices, probabilities))

    for index, state in enumerate(states):
        state.tick(
            now,
            angles[index],
            prob_by_index.get(index),
            is_sitting=sitting_flags[index],
        )


def _prune_stale(
    tracks: dict[int, PersonTrackState],
    frame_idx: int,
    max_inactive_frames: int,
) -> dict[int, PersonTrackState]:
    """Xoá trạng thái của những ID đã rời khỏi khung hình quá lâu."""
    return {
        track_id: state
        for track_id, state in tracks.items()
        if frame_idx - state.last_seen_frame <= max_inactive_frames
    }


def _has_people(results) -> bool:
    """Kết quả YOLO có chứa ít nhất một người đang được bám vết hay không."""
    if not results:
        return False
    boxes = results[0].boxes
    return (
        boxes is not None
        and boxes.id is not None
        and results[0].keypoints is not None
    )


def main() -> None:
    settings = load_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Khởi động hệ thống trên thiết bị: {device}")

    if not settings.telegram_enabled:
        print("⚠️  Chưa cấu hình TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID — cảnh báo chỉ in ra terminal.")

    pose_model = YOLO(settings.yolo_weights)
    lstm_model, seq_len, frame_stride = _load_lstm(settings, device)
    notifier = FallAlertNotifier(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        cooldown_sec=settings.alert_cooldown_sec,
    )

    capture = cv2.VideoCapture(settings.video_source)
    if not capture.isOpened():
        print(f"❌ Không thể mở nguồn video: {settings.video_source}")
        return

    tracks: dict[int, PersonTrackState] = {}
    frame_idx = 0
    previous_time = time.monotonic()

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break

        frame_idx += 1
        now = time.monotonic()
        annotated = frame

        results = pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=YOLO_CONFIDENCE,
            verbose=False,
        )

        if _has_people(results):
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints = results[0].keypoints.data.cpu().numpy()

            annotated = results[0].plot(boxes=False)

            states: list[PersonTrackState] = []
            for track_id in track_ids:
                state = tracks.get(int(track_id))
                if state is None:
                    state = PersonTrackState(seq_len=seq_len)
                    state.configure_verifier(settings.build_verifier())
                    tracks[int(track_id)] = state
                state.last_seen_frame = frame_idx
                states.append(state)

            _step_people(
                states,
                keypoints,
                frame_idx,
                frame_stride,
                settings,
                lstm_model,
                device,
                now,
            )

            for index, track_id in enumerate(track_ids):
                state = states[index]
                draw_person(annotated, boxes[index], int(track_id), state)
                if state.is_fallen:
                    notifier.trigger_alert(annotated, int(track_id), state.fall_prob)

        if frame_idx % STALE_CHECK_INTERVAL == 0:
            tracks = _prune_stale(tracks, frame_idx, settings.max_inactive_frames)

        fps = 1.0 / max(now - previous_time, 1e-5)
        previous_time = now
        draw_hud(annotated, fps, len(tracks))

        cv2.imshow(WINDOW_TITLE, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
