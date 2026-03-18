import os, base64, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque
from pathlib import Path
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Import your NLP Engine
from nlp_interference import glosstosentenceinference

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ── 1. Configuration & Setup ──────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'labels_csv': 'labels.csv',
    'model_path': 'best_fsl_model_final_0004.pth',  # Match the new 0004 model
    'num_frames': 48,
    'feature_dim': 198,
    'num_classes': 105,
    'd_model': 384,
    'nhead': 8,
    'num_layers': 6,
    'dim_ff': 768,
    'dropout': 0.25, # Matched to 0004 specs
    'top_k': 5,
    'pose_model': 'models/pose_landmarker.task',
    'hand_model': 'models/hand_landmarker.task',
    'face_model': 'models/face_landmarker.task',
    
    # EXACT APP.PY TIMING
    'record_secs': 3.0,
    'pre_buffer': 5,         # <--- SURGICAL FIX: Added pre-buffer control
    'auto_clear_secs': 4.0
}

CONFIDENCE_THRESH = 40.0

IDLE = 'IDLE'
SIGNING = 'SIGNING'
EVALUATE = 'EVALUATE'

FEATURE_DIM = 198
POSE_IDX = [11, 12, 13, 14, 15, 16]
FACE_IDX = [1, 33, 61, 199, 263, 291]

# Global State Machine Variables
state = IDLE
raw_rows = []
rolling_buffer = deque(maxlen=CONFIG['pre_buffer']) # <--- SURGICAL FIX: Bound to 5 frames

# Time trackers
rec_start_time = None
idle_start_time = time.time()
no_hand_count = 0 # <--- SURGICAL FIX: Added missing global tracker

gloss_buffer = []
translated_text = ""
last_prediction = ""
last_confidence = 0.0

# ── 2. Labels & Detectors ─────────────────────────────────────
try:
    labels_df = pd.read_csv(CONFIG['labels_csv'])
    class_names = [labels_df[labels_df['id'] == i]['label'].values[0] for i in range(CONFIG['num_classes'])]
    print(f'Loaded {len(class_names)} classes')
except Exception as e:
    print(f"❌ Could not load labels: {e}")
    exit()

_POSE = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=CONFIG['pose_model']),
        running_mode=mp_vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.4))

_HAND = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=CONFIG['hand_model']),
        running_mode=mp_vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.4))

_FACE = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=CONFIG['face_model']),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3))

print('Detectors initialized ✅')

# ── 3. Feature Pipeline ───────────────────────────────────────
def extract_frame_live(bgr_frame):
    row = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    row[144:180] = 0. 
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pr = _POSE.detect(img)
    hr = _HAND.detect(img)
    fr = _FACE.detect(img)

    if pr.pose_landmarks:
        lms = pr.pose_landmarks[0]
        for k, joint_idx in enumerate(POSE_IDX):
            row[k * 3: k * 3 + 3] = [lms[joint_idx].x, lms[joint_idx].y, lms[joint_idx].z]

    lc = np.full(63, np.nan, dtype=np.float32)
    rc = np.full(63, np.nan, dtype=np.float32)
    hand_ok = hr.hand_landmarks is not None and len(hr.hand_landmarks) > 0
    if hand_ok:
        for hl, hd in zip(hr.hand_landmarks, hr.handedness):
            coords = np.array([[l.x, l.y, l.z] for l in hl], dtype=np.float32).flatten()
            if hd[0].category_name == 'Left': lc = coords
            else: rc = coords
            
    row[18:81] = lc
    row[81:144] = rc
    
    if fr.face_landmarks:
        lms = fr.face_landmarks[0]
        for k, idx in enumerate(FACE_IDX): 
            row[180 + k * 3: 180 + k * 3 + 3] = [lms[idx].x, lms[idx].y, lms[idx].z]

    return row, hr, pr, fr, hand_ok

def raw_rows_to_skeleton(raw_rows):
    # ── EXACT 0004 PREPROCESSING CLONE ──
    seq = np.stack(raw_rows, axis=0).astype(np.float32)
    
    # 1. Force strict 48-frame timeline via resize
    seq = cv2.resize(seq, (198, CONFIG['num_frames']), interpolation=cv2.INTER_LINEAR)

    # 2. Impute & Low-Pass Filter (The Tremor Fix)
    df  = pd.DataFrame(seq)
    df  = df.interpolate(method='linear', axis=0).ffill().bfill().fillna(0.)
    df  = df.rolling(window=3, min_periods=1, center=True).mean()
    seq = df.values.astype(np.float32)
    
    # 3. Custom Spatial Normalization
    lx, ly = seq[:, 0], seq[:, 1]; rx, ry = seq[:, 3], seq[:, 4]
    midx, midy, midz = (lx + rx) / 2., (ly + ry) / 2., (seq[:, 2] + seq[:, 5]) / 2.
    sw = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2).astype(np.float32) + 1e-6
    
    if not (np.isnan(sw).all() or (sw < 1e-5).all()):
        for i in range(0, 18, 3): 
            seq[:, i] = (seq[:, i] - midx) / sw; seq[:, i + 1] = (seq[:, i + 1] - midy) / sw; seq[:, i + 2] = (seq[:, i + 2] - midz) / sw
        for start in [18, 81]:
            wx, wy, wz = seq[:, start].copy(), seq[:, start + 1].copy(), seq[:, start + 2].copy()
            hs = np.sqrt((wx - seq[:, start + 27]) ** 2 + (wy - seq[:, start + 28]) ** 2) + 1e-6
            for i in range(start, start + 63, 3): 
                seq[:, i] = (seq[:, i] - wx) / hs; seq[:, i + 1] = (seq[:, i + 1] - wy) / hs; seq[:, i + 2] = (seq[:, i + 2] - wz) / hs
        for i in range(180, 198, 3): 
            seq[:, i] = (seq[:, i] - midx) / sw; seq[:, i + 1] = (seq[:, i + 1] - midy) / sw; seq[:, i + 2] = (seq[:, i + 2] - midz) / sw
            
    seq = np.nan_to_num(seq, nan=0., posinf=0., neginf=0.)
    
    # 4. Kinematic Velocity
    T = seq.shape[0]
    pv, rv, lv = np.zeros((T, 18), dtype=np.float32), np.zeros((T, 9), dtype=np.float32), np.zeros((T, 9), dtype=np.float32)
    pv[1:], rv[1:], lv[1:] = seq[1:, :18] - seq[:-1, :18], seq[1:, 81:90] - seq[:-1, 81:90], seq[1:, 18:27] - seq[:-1, 18:27]
    pv[0], rv[0], lv[0] = pv[1], rv[1], lv[1]
    seq[:, 144:162], seq[:, 162:171], seq[:, 171:180] = pv, rv, lv
    
    return seq

# ── 4. PyTorch Model ──────────────────────────────────────────
class FSLTransformer(nn.Module):
    def __init__(self, feature_dim=198, d_model=384, nhead=8, num_layers=6, dim_ff=768, dropout=0.25, num_classes=105, seq_len=48):
        super().__init__()
        xp = (list(range(0, 18, 3)) + list(range(18, 81, 3)) + list(range(81, 144, 3)) + list(range(144, 180, 3)) + list(range(180, 198, 3)))
        yp = (list(range(1, 18, 3)) + list(range(19, 81, 3)) + list(range(82, 144, 3)) + list(range(145, 180, 3)) + list(range(181, 198, 3)))
        self.register_buffer('x_idx', torch.tensor(xp, dtype=torch.long))
        self.register_buffer('y_idx', torch.tensor(yp, dtype=torch.long))
        self.x_proj = nn.Linear(len(xp), d_model)
        self.y_proj = nn.Linear(len(yp), d_model)
        self.x_pos = nn.Embedding(seq_len, d_model)
        self.y_pos = nn.Embedding(seq_len, d_model)
        self.x_norm = nn.LayerNorm(d_model)
        self.y_norm = nn.LayerNorm(d_model)

        def enc():
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, activation='gelu', norm_first=True),
                num_layers=num_layers, enable_nested_tensor=False)

        self.x_enc = enc()
        self.y_enc = enc()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model * 2)
        
        self.proj_head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 128))
                                       
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d_model // 2, num_classes))

    def encode(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        xe = self.x_norm(self.x_proj(x[:, :, self.x_idx]) + self.x_pos(pos))
        ye = self.y_norm(self.y_proj(x[:, :, self.y_idx]) + self.y_pos(pos))
        xe = self.x_enc(xe)
        ye = self.y_enc(ye)
        fused, _ = self.cross_attn(query=xe, key=ye, value=ye)
        return self.pool_norm(torch.cat([fused, xe], dim=-1).mean(dim=1))

    def forward(self, x):
        return self.classifier(self.encode(x))

try:
    model = FSLTransformer(feature_dim=CONFIG['feature_dim'], d_model=CONFIG['d_model'], nhead=CONFIG['nhead'], num_layers=CONFIG['num_layers'], dim_ff=CONFIG['dim_ff'], dropout=CONFIG['dropout'], num_classes=CONFIG['num_classes'], seq_len=CONFIG['num_frames']).to(device)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device), strict=False)
    model.eval()
    print(f'Model loaded from {CONFIG["model_path"]} ✅  device={device}')
except Exception as e:
    print(f"❌ CRITICAL: Could not load PyTorch model: {e}")
    exit()

# EXACT 0004 INFERENCE (No TTA Noise)
def predict_sign(seq):
    tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    top_idx = probs.argsort()[::-1][:CONFIG['top_k']]
    return [(class_names[i], float(probs[i]) * 100) for i in top_idx]

# ── 5. API Routes ─────────────────────────────────────────────
def decode_base64_image(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/predict', methods=['POST'])
def predict():
    global state, raw_rows, rolling_buffer
    global rec_start_time, idle_start_time, no_hand_count # <--- SURGICAL FIX
    global gloss_buffer, translated_text, last_prediction, last_confidence

    now = time.time()

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' key with Base64 payload."}), 400

        frame = decode_base64_image(data['image'])
        
        raw_row, hr, pr, fr, hand_ok = extract_frame_live(frame)

        # ── Extract Lightweight Drawing Coordinates ──
        draw_points = []
        if pr.pose_landmarks:
            lms = pr.pose_landmarks[0]
            for idx in POSE_IDX: draw_points.append({'x': lms[idx].x, 'y': lms[idx].y, 'type': 'pose'})
        if hr.hand_landmarks:
            for idx, lms in enumerate(hr.hand_landmarks):
                side = hr.handedness[idx][0].category_name if idx < len(hr.handedness) else 'Right'
                for lm in lms: draw_points.append({'x': lm.x, 'y': lm.y, 'type': side})
        if fr.face_landmarks:
            lms = fr.face_landmarks[0]
            for idx in FACE_IDX: draw_points.append({'x': lms[idx].x, 'y': lms[idx].y, 'type': 'face'})

        # ── State Machine Logic ─────────────────────
        if state == IDLE:
            rolling_buffer.append(raw_row)
            if hand_ok:
                raw_rows = list(rolling_buffer)
                rec_start_time = now
                no_hand_count = 0 # <--- SURGICAL FIX
                state = SIGNING
                print('Auto-Triggered! Recording for 3s...')
            else:
                if len(gloss_buffer) > 0 and (now - idle_start_time) >= CONFIG['auto_clear_secs']:
                    gloss_buffer.clear()
                    translated_text = ""
                    print("\n[Auto-Cleared] 4 seconds of inactivity.")
                    idle_start_time = now

        elif state == SIGNING:
            raw_rows.append(raw_row)
            
            # <--- SURGICAL FIX: Track disappearing hands
            if hand_ok:
                no_hand_count = 0
            else:
                no_hand_count += 1
                
            # <--- SURGICAL FIX: Early exit & Trailing crop trigger
            if (now - rec_start_time) >= CONFIG['record_secs'] or no_hand_count >= 15:
                trim_n = min(no_hand_count, len(raw_rows) - 8)
                if trim_n > 0:
                    raw_rows = raw_rows[:-trim_n]
                
                state = EVALUATE

        # ── Evaluation Block ────────────────────────────────────
        if state == EVALUATE:
            seq = raw_rows_to_skeleton(raw_rows)
            display_top5 = predict_sign(seq)
            lbl, conf = display_top5[0]

            last_prediction = lbl
            last_confidence = conf

            if conf >= CONFIDENCE_THRESH:
                gloss_buffer.append(lbl.lower().replace(" ", "_"))
                gloss_string = " ".join(gloss_buffer)
                translated_text = glosstosentenceinference(gloss_string)
                print(f"✅ Translated: '{translated_text}'")

            raw_rows = []
            rolling_buffer.clear()
            idle_start_time = now  
            state = IDLE

        return jsonify({
            "status": "success",
            "state": state,
            "new_sign": last_prediction if last_confidence >= CONFIDENCE_THRESH else "...",
            "history": " ".join(gloss_buffer).replace("_", " "),
            "sentence": translated_text if translated_text else "...",
            "confidence": last_confidence,
            "landmarks": draw_points
        })

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)