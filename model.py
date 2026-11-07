import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 30
FEATURE_DIM = 52
FRAME_STRIDE = 2

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_dim=52, hidden_dim=128, num_layers=2, dropout=0.3, seq_len=30):
        super(FallDetectionLSTM, self).__init__()

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(seq_len),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch, 30, 52)
        x = self.feature_proj(x)
        lstm_out, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        logits = self.classifier(context)
        return logits.squeeze(-1)


def normalize_single_frame_keypoints(kpts_17x3):
    """
    Chuẩn hóa 17 keypoints (17, 3) của 1 người thành vector 52 chiều.
    Trả về thêm số lượng khớp hợp lệ (valid_count) để lọc nhiễu khi bị che khuất.
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    x, y, conf = kpts_17x3[:, 0], kpts_17x3[:, 1], kpts_17x3[:, 2]

    valid_mask = (conf >= 0.2) & (x > 0) & (y > 0)
    valid_count = int(np.sum(valid_mask))
    if valid_count < 4:
        return feat, valid_count, 0.0

    x_min, x_max = np.min(x[valid_mask]), np.max(x[valid_mask])
    y_min, y_max = np.min(y[valid_mask]), np.max(y[valid_mask])

    w = max(x_max - x_min, 1e-5)
    h = max(y_max - y_min, 1e-5)
    aspect_ratio = float(np.clip(w / h, 0.0, 5.0))

    norm_x = np.where(valid_mask, np.clip((x - x_min) / w, 0.0, 1.0), 0.0)
    norm_y = np.where(valid_mask, np.clip((y - y_min) / h, 0.0, 1.0), 0.0)
    clean_conf = np.where(valid_mask, conf, 0.0)

    feat[:51] = np.stack([norm_x, norm_y, clean_conf], axis=-1).flatten()
    feat[51] = aspect_ratio
    return feat, valid_count, aspect_ratio