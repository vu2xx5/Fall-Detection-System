"""Cấu hình hệ thống.

Toàn bộ tham số được nạp từ biến môi trường hoặc file `.env` đặt cạnh mã nguồn,
nhờ vậy không có thông tin nhạy cảm nào (token Telegram, chat ID) bị commit lên Git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".env"

DEFAULT_VIDEO_SOURCE = "0"
DEFAULT_LSTM_WEIGHTS = "best_fall_lstm.pth"
DEFAULT_YOLO_WEIGHTS = "yolov8n-pose.pt"
DEFAULT_ALERT_COOLDOWN_SEC = 60
DEFAULT_FALL_PROB_THRESHOLD = 0.75
DEFAULT_FALL_STREAK_REQUIRED = 3
DEFAULT_MIN_VALID_KEYPOINTS = 6
DEFAULT_MAX_INACTIVE_FRAMES = 150


def load_env_file(path: Path = ENV_FILE) -> None:
    """Nạp các cặp KEY=VALUE từ file .env vào os.environ.

    Biến môi trường đã tồn tại luôn được ưu tiên hơn giá trị trong file,
    nên có thể ghi đè cấu hình khi chạy mà không cần sửa .env.
    """
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    """Tập tham số bất biến của hệ thống."""

    video_source: str | int
    lstm_weights: str
    yolo_weights: str
    telegram_bot_token: str
    telegram_chat_id: str
    alert_cooldown_sec: int
    fall_prob_threshold: float
    fall_streak_required: int
    min_valid_keypoints: int
    max_inactive_frames: int

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


def _parse_video_source(raw: str) -> str | int:
    """'0' -> webcam số 0, còn lại giữ nguyên (đường dẫn file hoặc URL RTSP)."""
    return int(raw) if raw.lstrip("-").isdigit() else raw


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def load_settings() -> Settings:
    """Đọc cấu hình từ biến môi trường (sau khi đã nạp file .env nếu có)."""
    load_env_file()

    return Settings(
        video_source=_parse_video_source(_env_str("VIDEO_SOURCE", DEFAULT_VIDEO_SOURCE)),
        lstm_weights=_env_str("LSTM_WEIGHTS", DEFAULT_LSTM_WEIGHTS),
        yolo_weights=_env_str("YOLO_WEIGHTS", DEFAULT_YOLO_WEIGHTS),
        telegram_bot_token=_env_str("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=_env_str("TELEGRAM_CHAT_ID", ""),
        alert_cooldown_sec=_env_int("ALERT_COOLDOWN_SEC", DEFAULT_ALERT_COOLDOWN_SEC),
        fall_prob_threshold=_env_float("FALL_PROB_THRESHOLD", DEFAULT_FALL_PROB_THRESHOLD),
        fall_streak_required=_env_int("FALL_STREAK_REQUIRED", DEFAULT_FALL_STREAK_REQUIRED),
        min_valid_keypoints=_env_int("MIN_VALID_KEYPOINTS", DEFAULT_MIN_VALID_KEYPOINTS),
        max_inactive_frames=_env_int("MAX_INACTIVE_FRAMES", DEFAULT_MAX_INACTIVE_FRAMES),
    )
