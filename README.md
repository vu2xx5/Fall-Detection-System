# 🚨 Fall Detection System — Phát hiện té ngã đa đối tượng

Hệ thống thị giác máy tính phát hiện **té ngã theo thời gian thực** cho **nhiều người cùng lúc** trong khung hình, kết hợp:

- **YOLOv8-Pose** + **ByteTrack** → nhận diện khung xương 17 điểm và bám vết từng người bằng ID riêng.
- **Bi-LSTM + Attention** (PyTorch) → phân tích chuỗi chuyển động 30 frame để phân loại *Normal* / *Fall*.
- **Telegram Bot** → gửi ảnh chụp + thông tin cảnh báo về điện thoại ngay khi phát hiện ngã.

> Ứng dụng thực tế: giám sát người cao tuổi tại nhà, bệnh viện, phòng chờ, khu vực làm việc một mình — nơi cần cảnh báo tức thời khi có người ngã mà không có ai chứng kiến.

---

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#-cấu-hình)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cơ chế chống báo giả](#-cơ-chế-chống-báo-giả)
- [Thiết lập Telegram Bot](#-thiết-lập-telegram-bot)
- [Xử lý sự cố](#-xử-lý-sự-cố)
- [Hạn chế & hướng phát triển](#-hạn-chế--hướng-phát-triển)

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| **Đa đối tượng** | Theo dõi và dự đoán độc lập cho từng người nhờ ByteTrack, không lẫn trạng thái giữa các ID |
| **Batch inference** | Tất cả người đủ 30 frame được đưa vào model trong **một lần forward** duy nhất → tăng FPS |
| **Chống báo giả 3 lớp** | Ngưỡng xác suất + chuỗi dự đoán liên tiếp + lọc theo số keypoint nhìn rõ |
| **Cảnh báo bất đồng bộ** | Gửi Telegram chạy trong thread riêng, không chặn vòng lặp video |
| **Cooldown theo ID** | Mỗi người chỉ gửi tối đa 1 cảnh báo / 60 giây, tránh spam |
| **Đa nguồn video** | Webcam, file video, hoặc luồng RTSP từ camera IP |
| **Cấu hình tách rời** | Toàn bộ tham số và token nằm trong `.env`, không hardcode trong mã nguồn |

---

## 🏗 Kiến trúc hệ thống

```
                    ┌──────────────────────────────┐
   Webcam / Video   │        cv2.VideoCapture      │
   RTSP Camera ────▶│        (từng frame BGR)      │
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │   YOLOv8n-Pose + ByteTrack   │
                    │  • 17 keypoints / người      │
                    │  • track_id ổn định          │
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │  Chuẩn hoá keypoints → 52-dim│
                    │  (34 toạ độ + 17 conf + AR)  │
                    │  • Lọc theo MIN_VALID_KEYPOINTS
                    │  • Lấy mẫu mỗi FRAME_STRIDE  │
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │  Buffer 30 frame / track_id  │
                    │  (deque, riêng cho mỗi ID)   │
                    └───────────────┬──────────────┘
                                    ▼  đủ 30 frame
                    ┌──────────────────────────────┐
                    │  Bi-LSTM (2 lớp) + Attention │
                    │  → sigmoid → P(ngã) ∈ [0,1]  │
                    └───────────────┬──────────────┘
                                    ▼
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
      ┌────────────────────┐              ┌────────────────────────┐
      │ Vẽ bbox + trạng thái│              │ Cảnh báo Telegram      │
      │ Xanh: Normal        │              │ (thread riêng, cooldown)│
      │ Đỏ:   FALL!         │              │ ảnh + ID + % + giờ     │
      └────────────────────┘              └────────────────────────┘
```

### Vector đặc trưng 52 chiều

Mỗi frame của một người được `normalize_single_frame_keypoints()` chuyển thành vector 52 chiều, bất biến với vị trí và khoảng cách tới camera:

| Thành phần | Số chiều | Ý nghĩa |
|---|---|---|
| `norm_x[17]` | 17 | Toạ độ X chuẩn hoá theo bounding box khung xương |
| `norm_y[17]` | 17 | Toạ độ Y chuẩn hoá |
| `conf[17]` | 17 | Độ tin cậy của từng keypoint |
| `aspect_ratio` | 1 | Tỉ lệ rộng/cao của khung xương — tín hiệu mạnh khi người đổ ngang |

`aspect_ratio` tăng vọt khi người chuyển từ tư thế đứng sang nằm/ngã, nên nó là đặc trưng quyết định giúp model phân biệt ngã với cúi người.

### Mô hình `FallDetectionLSTM`

```
Input (B, 30, 52)
   └─ Linear(52→128) → BatchNorm1d → SiLU → Dropout(0.3)
   └─ Bi-LSTM(hidden=128, layers=2, bidirectional=True)   → (B, 30, 256)
   └─ Attention(256→64→1) + softmax theo trục thời gian
   └─ context = Σ(attn_weight × lstm_out)                 → (B, 256)
   └─ Linear(256→64) → BatchNorm1d → SiLU → Dropout
   └─ Linear(64→1) → sigmoid                              → P(ngã)
```

Cơ chế **Attention** cho phép model tập trung vào đúng những frame quyết định (khoảnh khắc mất thăng bằng) thay vì chia đều trọng số cho cả 30 frame.

---

## ⚙️ Cài đặt

### Yêu cầu

- **Python 3.10 – 3.13**
- **ffmpeg** (khuyến nghị, để OpenCV đọc được nhiều định dạng video)
- Webcam hoặc camera IP (RTSP), hoặc một file video có sẵn
- GPU NVIDIA (tuỳ chọn) — hệ thống tự phát hiện CUDA, chạy CPU vẫn được nhưng FPS thấp hơn

### Các bước

```bash
# 1. Clone dự án
git clone https://github.com/<username>/Fall-Detection-System.git
cd Fall-Detection-System

# 2. Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Tạo file cấu hình từ mẫu
cp .env.example .env
#    → mở .env và điền TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (nếu muốn nhận cảnh báo)

# 5. Chạy
python main.py
```

> **Lưu ý về trọng số mô hình:** repo đã kèm sẵn `best_fall_lstm.pth` (~2.7 MB) và `yolov8n-pose.pt` (~6.5 MB). Nếu thiếu `yolov8n-pose.pt`, thư viện `ultralytics` sẽ **tự động tải** khi chạy lần đầu.

### Cài PyTorch có hỗ trợ GPU

`requirements.txt` cài bản PyTorch mặc định (CPU trên macOS, CUDA trên Linux/Windows). Nếu cần chỉ định rõ phiên bản CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 🔧 Cấu hình

Mọi tham số nằm trong file `.env` (được `.gitignore` bỏ qua). Sau khi `cp .env.example .env`, chỉnh các biến sau:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VIDEO_SOURCE` | `0` | `0` = webcam mặc định · `test.mp4` = file video · `rtsp://user:pass@ip:554/...` = camera IP |
| `LSTM_WEIGHTS` | `best_fall_lstm.pth` | Đường dẫn file trọng số Bi-LSTM |
| `YOLO_WEIGHTS` | `yolov8n-pose.pt` | Đường dẫn file trọng số YOLOv8-Pose |
| `TELEGRAM_BOT_TOKEN` | *(trống)* | Token bot lấy từ [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | *(trống)* | Chat ID nhận cảnh báo, lấy từ [@userinfobot](https://t.me/userinfobot) |
| `ALERT_COOLDOWN_SEC` | `60` | Mỗi ID chỉ gửi tối đa 1 cảnh báo trong khoảng này (giây) |
| `FALL_PROB_THRESHOLD` | `0.75` | Ngưỡng xác suất để coi một dự đoán là "ngã" |
| `FALL_STREAK_REQUIRED` | `3` | Số dự đoán "ngã" **liên tiếp** bắt buộc trước khi cảnh báo |
| `MIN_VALID_KEYPOINTS` | `6` | Số keypoint tối thiểu nhìn rõ (trên 17) để dùng frame đó |
| `MAX_INACTIVE_FRAMES` | `150` | Xoá trạng thái ID nếu biến mất quá số frame này (~5 giây ở 30 FPS) |

Nếu `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` để trống, hệ thống **vẫn chạy bình thường** và chỉ in cảnh báo ra terminal — rất tiện để thử nghiệm offline.

Biến môi trường của shell luôn được ưu tiên hơn giá trị trong `.env`, nên có thể ghi đè nhanh:

```bash
VIDEO_SOURCE=test.mp4 FALL_PROB_THRESHOLD=0.6 python main.py
```

---

## ▶️ Sử dụng

```bash
python main.py
```

Cửa sổ hiển thị kết quả với:

- **Khung xương 17 điểm** do YOLOv8-Pose vẽ.
- **Bounding box xanh lá** — người bình thường, kèm nhãn `Collecting (x/30)` khi còn đang tích luỹ frame, hoặc `Normal (xx%)` khi đã có dự đoán.
- **Bounding box đỏ + nhãn `FALL! (xx%)`** — phát hiện té ngã; cảnh báo Telegram được kích hoạt ngay.
- **Góc trên trái:** FPS hiện tại và tổng số người đang được theo dõi.

Nhấn **`q`** để thoát.

### Ví dụ kết quả trên terminal

```
🚀 Khởi động hệ thống trên thiết bị: mps
✅ Đã nạp thành công best_fall_lstm.pth (Seq_Len=30, Stride=2)
✅ Đã gửi cảnh báo Telegram cho ID #3
```

---

## 📁 Cấu trúc dự án

```
Fall_Detection_System/
├── main.py                  # Vòng lặp chính: đọc video → pose → LSTM → vẽ & cảnh báo
├── model.py                 # Kiến trúc Bi-LSTM + Attention và hàm trích xuất 52-dim
├── notifier.py              # Gửi cảnh báo Telegram (thread riêng + cooldown theo ID)
├── config.py                # Nạp cấu hình từ biến môi trường / file .env
├── best_fall_lstm.pth       # Trọng số Bi-LSTM đã huấn luyện
├── yolov8n-pose.pt          # Trọng số YOLOv8-Pose (17 keypoints)
├── requirements.txt         # Thư viện phụ thuộc
├── .env.example             # Mẫu cấu hình (được commit)
├── .env                     # Cấu hình thật (KHÔNG commit)
├── .gitignore
└── README.md
```

| File | Trách nhiệm |
|---|---|
| `main.py` | Điều phối toàn bộ pipeline, quản lý trạng thái từng `track_id`, batch inference, vẽ overlay |
| `model.py` | Định nghĩa `FallDetectionLSTM`, hằng số `SEQ_LEN` / `FEATURE_DIM` / `FRAME_STRIDE`, hàm chuẩn hoá keypoint |
| `notifier.py` | `FallAlertNotifier` — cooldown theo ID, encode JPEG, POST lên Telegram API trong daemon thread |
| `config.py` | `Settings` (frozen dataclass) + `load_settings()`, đọc `.env` không cần thư viện ngoài |

---

## 🛡 Cơ chế chống báo giả

Báo động giả là vấn đề lớn nhất của hệ thống fall detection. Dự án áp dụng **3 lớp lọc độc lập**:

**Lớp 1 — Lọc chất lượng dữ liệu vào.** Một frame chỉ được đưa vào buffer nếu có ít nhất `MIN_VALID_KEYPOINTS` (mặc định 6/17) khớp xương nhìn rõ (`conf ≥ 0.2`). Khi người bị khuất sau bàn, cột, hay ra khỏi rìa khung hình, dữ liệu rác không làm nhiễu chuỗi.

**Lớp 2 — Lọc theo ngưỡng xác suất.** Chỉ dự đoán có `P(ngã) ≥ FALL_PROB_THRESHOLD` (mặc định `0.75`) mới được tính là một "phiếu ngã".

**Lớp 3 — Lọc theo thời gian (streak).** Cần `FALL_STREAK_REQUIRED` (mặc định 3) dự đoán ngã **liên tiếp** mới kích hoạt cảnh báo. Mỗi dự đoán không-ngã sẽ giảm streak đi 1 (`max(0, streak - 1)`), nên một cú nhảy, cúi xuống nhặt đồ, hay ngồi xuống ghế nhanh sẽ không đủ để tạo chuỗi.

Ngoài ra, `ALERT_COOLDOWN_SEC` (mặc định 60s) giới hạn tần suất gửi tin theo từng `track_id`, nên người đã ngã và đang nằm dưới sàn sẽ không làm ngập hộp thư Telegram.

> **Tinh chỉnh:** môi trường nhiều báo giả → tăng `FALL_PROB_THRESHOLD` lên `0.85` hoặc `FALL_STREAK_REQUIRED` lên `5`. Hệ thống phản ứng chậm → giảm `FALL_STREAK_REQUIRED` xuống `2`, nhưng đổi lại độ chính xác giảm.

---

## 📱 Thiết lập Telegram Bot

1. Mở Telegram, tìm **[@BotFather](https://t.me/BotFather)** → gửi `/newbot` → đặt tên và username cho bot.
2. BotFather trả về **token** dạng `1234567890:AAH...` → dán vào `TELEGRAM_BOT_TOKEN`.
3. Tìm **[@userinfobot](https://t.me/userinfobot)** → gửi bất kỳ tin nhắn → bot trả về **`Id`** của bạn → dán vào `TELEGRAM_CHAT_ID`.
4. **Quan trọng:** mở chat với bot vừa tạo và bấm **Start** ít nhất một lần. Telegram không cho bot chủ động nhắn tin cho người chưa khởi động chat với nó.

Muốn gửi cảnh báo vào một **nhóm**: thêm bot vào nhóm, gửi một tin nhắn bất kỳ trong nhóm, rồi dùng `chat_id` âm (dạng `-100xxxxxxxxx`) lấy từ `getUpdates`:

```bash
curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
```

Kiểm tra nhanh token có hợp lệ không:

```bash
curl "https://api.telegram.org/bot<TOKEN>/getMe"
```

---

## 🩺 Xử lý sự cố

| Triệu chứng | Nguyên nhân & cách khắc phục |
|---|---|
| `❌ Không thể mở nguồn video` | `VIDEO_SOURCE` sai. Trên macOS cần cấp quyền Camera cho Terminal/VS Code trong *System Settings → Privacy & Security → Camera*. |
| `FileNotFoundError: best_fall_lstm.pth` | Chạy `python main.py` từ sai thư mục. Phải `cd` vào thư mục gốc dự án. |
| `⚠️ Không thể kết nối Telegram API` | Sai token, chưa bấm Start với bot, hoặc máy không có Internet. Kiểm tra bằng `getMe`. |
| Không thấy bbox nào | Người ở quá xa, ngược sáng, hoặc bị che khuất. Hạ `conf` trong `pose_model.track(...)` (mặc định `0.4`). |
| Báo ngã liên tục khi không ai ngã | Tăng `FALL_PROB_THRESHOLD` và `FALL_STREAK_REQUIRED`; kiểm tra lại `aspect_ratio` có bị nhiễu do bbox sai. |
| FPS quá thấp | Chạy CPU. Cài PyTorch bản CUDA/MPS, hoặc giảm độ phân giải đầu vào, hoặc tăng `FRAME_STRIDE` (lấy mẫu thưa hơn). |
| `IndexError` ở `boxes_xyxy[i]` | Bounding box và keypoints lệch số lượng — thường do tracker trả về `id` rỗng. Khởi động lại tiến trình. |

---

## ⚠️ Hạn chế & hướng phát triển

**Hạn chế hiện tại**

- Độ trễ khởi động: một người mới vào khung hình cần **30 frame hữu ích** (~2 giây ở 30 FPS với stride 2) trước khi có dự đoán đầu tiên.
- Nhạy cảm với góc nhìn: model huấn luyện trên góc máy ngang; góc từ trên cao (top-down) cho độ chính xác thấp hơn.
- Một người bị đổi `track_id` giữa chừng (bị người khác che khuất lâu) sẽ phải tích luỹ lại buffer từ đầu.
- Chưa có giao diện web; kết quả chỉ hiển thị trên cửa sổ OpenCV.

**Hướng phát triển**

- Thêm module huấn luyện (`train.py`) và script chuẩn bị dataset từ các bộ như UR Fall / Le2i / Multiple Cameras Fall.
- Xác thực qua web (FastAPI + WebSocket) để xem nhiều camera từ xa.
- Lưu lịch sử cảnh báo vào SQLite kèm ảnh và thời điểm để tra cứu sau.
- Xuất ONNX / TensorRT để chạy trên thiết bị biên (Jetson, Raspberry Pi + Coral).

---

## 📄 Giấy phép

Dự án phục vụ mục đích học tập và nghiên cứu. Trọng số `yolov8n-pose.pt` thuộc [Ultralytics](https://github.com/ultralytics/ultralytics) và tuân theo giấy phép **AGPL-3.0** của họ.

---

## 🙏 Ghi nhận

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — nhận diện tư thế và ByteTrack.
- [PyTorch](https://pytorch.org/) — huấn luyện và suy luận Bi-LSTM.
- [OpenCV](https://opencv.org/) — đọc/ghi video và hiển thị.
