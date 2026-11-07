import time
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import load_settings
from model import FallDetectionLSTM, normalize_single_frame_keypoints, SEQ_LEN, FEATURE_DIM, FRAME_STRIDE
from notifier import FallAlertNotifier

# Toàn bộ tham số (nguồn video, đường dẫn trọng số, token Telegram, các ngưỡng lọc)
# được nạp từ biến môi trường hoặc file .env — xem config.py và .env.example

class PersonTrackState:
    """Quản lý lịch sử 30 frames và trạng thái của từng người (track_id)"""
    def __init__(self, seq_len=30):
        self.buffer = deque(maxlen=seq_len)
        self.fall_prob = 0.0
        self.fall_streak = 0
        self.is_fallen = False
        self.last_seen_frame = 0

def main():
    settings = load_settings()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Khởi động hệ thống trên thiết bị: {device}")

    if not settings.telegram_enabled:
        print("⚠️  Chưa cấu hình TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID — cảnh báo chỉ in ra terminal.")

    # 1. Load mô hình YOLOv8-Pose
    pose_model = YOLO(settings.yolo_weights)

    # 2. Load mô hình Bi-LSTM + Attention từ file trọng số
    checkpoint = torch.load(settings.lstm_weights, map_location=device)
    seq_len = checkpoint.get('seq_len', SEQ_LEN)
    feature_dim = checkpoint.get('feature_dim', FEATURE_DIM)
    frame_stride = checkpoint.get('frame_stride', FRAME_STRIDE)

    lstm_model = FallDetectionLSTM(input_dim=feature_dim, seq_len=seq_len).to(device)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    print(f"✅ Đã nạp thành công {settings.lstm_weights} (Seq_Len={seq_len}, Stride={frame_stride})")

    # 3. Khởi tạo bộ gửi thông báo Telegram và Quản lý đa đối tượng
    notifier = FallAlertNotifier(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        cooldown_sec=settings.alert_cooldown_sec,
    )
    tracks = {}  # Dict: {track_id: PersonTrackState}

    cap = cv2.VideoCapture(settings.video_source)
    if not cap.isOpened():
        print(f"❌ Không thể mở nguồn video: {settings.video_source}")
        return

    frame_idx = 0
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 4. Chạy YOLOv8-Pose + ByteTrack để bám theo nhiều người
        results = pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            verbose=False
        )

        ready_track_ids = []
        batch_sequences = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints_data = results[0].keypoints.data.cpu().numpy() # (N, 17, 3)

            # Duyệt qua từng người trong khung hình
            for i, track_id in enumerate(track_ids):
                if track_id not in tracks:
                    tracks[track_id] = PersonTrackState(seq_len=seq_len)

                state = tracks[track_id]
                state.last_seen_frame = frame_idx

                # Chỉ trích xuất và đẩy vào hàng đợi theo đúng bước nhảy FRAME_STRIDE (giống lúc Train)
                if frame_idx % frame_stride == 0:
                    feat, valid_kpts, aspect_ratio = normalize_single_frame_keypoints(keypoints_data[i])

                    # Chỉ cập nhật nếu nhìn thấy đủ khớp xương (tránh bị che khuất)
                    if valid_kpts >= settings.min_valid_keypoints:
                        state.buffer.append(feat)

                    # Nếu người này đã tích lũy đủ 30 frames thì gom vào Batch để dự đoán
                    if len(state.buffer) == seq_len:
                        ready_track_ids.append(track_id)
                        batch_sequences.append(np.array(state.buffer, dtype=np.float32))

            # 5. Chạy dự đoán Bi-LSTM cho tất cả mọi người cùng lúc (Batch Inference)
            if len(batch_sequences) > 0:
                batch_tensor = torch.tensor(np.array(batch_sequences), dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits = lstm_model(batch_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    if probs.ndim == 0:
                        probs = np.array([probs])

                for tid, prob in zip(ready_track_ids, probs):
                    st = tracks[tid]
                    st.fall_prob = float(prob)

                    # Bộ lọc chống báo giả: kiểm tra số lần dự đoán liên tiếp
                    if st.fall_prob >= settings.fall_prob_threshold:
                        st.fall_streak += 1
                    else:
                        st.fall_streak = max(0, st.fall_streak - 1)

                    st.is_fallen = (st.fall_streak >= settings.fall_streak_required)

            # 6. Vẽ khung xương, Bounding Box cho từng người và Gửi cảnh báo nếu ngã
            annotated_frame = results[0].plot(boxes=False) # Chỉ vẽ khung xương từ YOLO

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = boxes_xyxy[i]
                st = tracks[track_id]

                if st.is_fallen:
                    color = (0, 0, 255) # Đỏ: Phát hiện ngã
                    label = f"ID #{track_id} | FALL! ({st.fall_prob*100:.0f}%)"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated_frame, label, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Kích hoạt gửi thông báo kèm ảnh về điện thoại
                    notifier.trigger_alert(annotated_frame, track_id, st.fall_prob)
                else:
                    color = (0, 255, 0) # Xanh lá: Bình thường
                    buf_progress = len(st.buffer)
                    if buf_progress < seq_len:
                        label = f"ID #{track_id} | Collecting ({buf_progress}/{seq_len})"
                    else:
                        label = f"ID #{track_id} | Normal ({st.fall_prob*100:.0f}%)"

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            annotated_frame = frame

        # 7. Dọn dẹp bộ nhớ các track_id đã rời khỏi camera
        if frame_idx % 60 == 0:
            stale_ids = [
                tid for tid, st in tracks.items()
                if frame_idx - st.last_seen_frame > settings.max_inactive_frames
            ]
            for tid in stale_ids:
                del tracks[tid]

        # Hiển thị tốc độ FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} | People: {len(tracks)}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Multi-Person Fall Detection System (Bi-LSTM + YOLOv8-Pose)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
