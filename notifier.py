import time
import threading
from datetime import datetime
import cv2
import requests

class FallAlertNotifier:
    def __init__(self, bot_token: str, chat_id: str, cooldown_sec: int = 60):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_sec = cooldown_sec
        self.last_alert_time = {} # Lưu thời điểm báo gần nhất của từng track_id

    def can_send_alert(self, track_id: int) -> bool:
        now = time.time()
        last_time = self.last_alert_time.get(track_id, 0)
        if now - last_time >= self.cooldown_sec:
            self.last_alert_time[track_id] = now
            return True
        return False

    def _send_telegram_worker(self, frame_bgr, track_id: int, prob: float):
        """Hàm chạy ngầm (Background Thread) gửi ảnh và thông báo"""
        timestamp = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
        caption = (
            f"🚨 CẢNH BÁO PHÁT HIỆN NGÃ!\n"
            f"• Đối tượng: ID #{track_id}\n"
            f"• Độ tin cậy AI: {prob * 100:.1f}%\n"
            f"• Thời gian: {timestamp}"
        )

        # Nếu chưa điền Token Telegram thì chỉ in log ra Terminal để test
        if not self.bot_token or self.bot_token == "YOUR_TELEGRAM_BOT_TOKEN":
            print(f"\n[ALERT LOG] {caption}\n")
            return

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            success, encoded_img = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                return

            files = {'photo': ('fall_alert.jpg', encoded_img.tobytes(), 'image/jpeg')}
            data = {'chat_id': self.chat_id, 'caption': caption}
            response = requests.post(url, data=data, files=files, timeout=10)

            if response.status_code == 200:
                print(f"✅ Đã gửi cảnh báo Telegram cho ID #{track_id}")
            else:
                print(f"⚠️ Lỗi gửi Telegram: {response.text}")
        except Exception as e:
            print(f"⚠️ Không thể kết nối Telegram API: {e}")

    def trigger_alert(self, frame_bgr, track_id: int, prob: float):
        """Gọi hàm này khi phát hiện ngã: kiểm tra cooldown và tạo luồng gửi ngầm"""
        if self.can_send_alert(track_id):
            frame_copy = frame_bgr.copy()
            thread = threading.Thread(
                target=self._send_telegram_worker,
                args=(frame_copy, track_id, prob),
                daemon=True
            )
            thread.start()