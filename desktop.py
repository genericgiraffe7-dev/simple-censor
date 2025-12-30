import sys
import os
import time
import ctypes
import numpy as np
import cv2
import mss
from ultralytics import YOLO

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QImage, QColor, QPen

# --- CONFIG ---
TARGET_MODEL = "best.onnx"
CONFIDENCE_THRESHOLD = 0.15  # 15% Sensitivity
UPDATE_DELAY = 10 
SMOOTHING_FACTOR = 0.5  
MAX_MISSING_FRAMES = 8 

# --- THE SWEET SPOT FOR 1366x768 ---
# We process at 640 pixels high.
# On your screen, this is almost 1:1 quality (very sharp).
PROCESSING_HEIGHT = 640

# --- VISUALS ---
STYLE_MODE = 1          # Pixelate
INTENSITY_VAL = 40      # Heavy

# --- EXPANSION ---
GLOBAL_PADDING = 0.15       
GENITAL_SCALE = 3.0         # 300% Radius for Vagina

# --- BLOCK LIST ---
BLOCK_LIST = [
    "Breasts (Exposed)", 
    "Breasts (Covered)",
    "Genitalia (Female Exposed)",
    "Genitalia (Female Covered)",
    "Buttocks (Exposed)",
    "Anus (Exposed)"
]

GENITALIA_CLASSES = ["Genitalia (Female Exposed)", "Genitalia (Female Covered)"]

# --- DPI FIX ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

# --- SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
model_path = os.path.join(MODELS_DIR, TARGET_MODEL)

if not os.path.exists(model_path):
    print(f"❌ CRITICAL: {TARGET_MODEL} missing.")
    sys.exit()

print(f"⏳ Loading {TARGET_MODEL}...")
try:
    model = YOLO(model_path, task='detect')
    print(f"✅ Model Loaded. Optimized for 1366x768 (Running at 640p).")
except Exception as e:
    print(f"❌ Model Error: {e}")
    sys.exit()

# --- MERGE LOGIC ---
def boxes_intersect(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    if (x1 > x2 + w2) or (x2 > x1 + w1): return False
    if (y1 > y2 + h2) or (y2 > y1 + h1): return False
    return True

def merge_boxes_union(boxes):
    if not boxes: return []
    while True:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            bx, by, bw, bh = boxes[i]
            x_min, y_min, x_max, y_max = bx, by, bx + bw, by + bh
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                if boxes_intersect((bx, by, bw, bh), boxes[j]):
                    ox, oy, ow, oh = boxes[j]
                    x_min = min(x_min, ox)
                    y_min = min(y_min, oy)
                    x_max = max(x_max, ox + ow)
                    y_max = max(y_max, oy + oh)
                    used[j] = True
                    merged = True 
            new_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        boxes = new_boxes
        if not merged: break
    return boxes

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

class StickyBox:
    def __init__(self, rect):
        self.x, self.y, self.w, self.h = rect
        self.missing_count = 0

    def update(self, new_rect):
        nx, ny, nw, nh = new_rect
        alpha = SMOOTHING_FACTOR
        self.x = int(alpha * nx + (1 - alpha) * self.x)
        self.y = int(alpha * ny + (1 - alpha) * self.y)
        self.w = int(alpha * nw + (1 - alpha) * self.w)
        self.h = int(alpha * nh + (1 - alpha) * self.h)
        self.missing_count = 0

    def get_rect(self):
        return (self.x, self.y, self.w, self.h)

class CensorOverlay(QWidget):
    def __init__(self):
        super().__init__()
        screen = QApplication.primaryScreen()
        rect = screen.geometry()
        self.setGeometry(rect)
        self.w = rect.width()
        self.h = rect.height()
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_PaintOnScreen)

        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
            print("✅ 'Invisibility Cloak' Enabled.")
        except Exception as e:
            print(f"⚠️ Warning: Could not set display affinity: {e}")

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.tracks = [] 
        self.render_list = [] 
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(UPDATE_DELAY)
        self.show()

    def apply_effect(self, roi, w, h):
        if w <= 0 or h <= 0: return roi
        if STYLE_MODE == 1: 
            block = INTENSITY_VAL
            if w < 60: block = max(2, block // 3)
            small = cv2.resize(roi, (max(1, w//block), max(1, h//block)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return roi

    def update_loop(self):
        try:
            raw_screen = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2BGR)
            
            # --- RESIZE TO 640p ---
            # At your 1366x768 resolution, this is nearly native quality.
            target_h = PROCESSING_HEIGHT
            scale = target_h / frame.shape[0]
            target_w = int(frame.shape[1] * scale)
            small_frame = cv2.resize(frame, (target_w, target_h))

            # Single Pass Inference (Fast & Sharp)
            results = model(small_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            raw_boxes = [] 

            for res in results:
                if not res.boxes: continue
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    try:
                        raw_name = model.names[cls_id]
                        LABEL_MAP = {
                            "FEMALE_BREAST_EXPOSED": "Breasts (Exposed)", 
                            "FEMALE_BREAST_COVERED": "Breasts (Covered)",
                            "FEMALE_GENITALIA_EXPOSED": "Genitalia (Female Exposed)", 
                            "FEMALE_GENITALIA_COVERED": "Genitalia (Female Covered)",
                            "BUTTOCKS_EXPOSED": "Buttocks (Exposed)", 
                            "ANUS_EXPOSED": "Anus (Exposed)"
                        }
                        friendly = LABEL_MAP.get(raw_name, raw_name)
                    except:
                        friendly = "unknown"

                    if friendly in BLOCK_LIST:
                        # Map from 640p -> 768p
                        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0])
                        
                        x1 = int(lx1 / scale)
                        y1 = int(ly1 / scale)
                        x2 = int(lx2 / scale)
                        y2 = int(ly2 / scale)
                        w, h = x2 - x1, y2 - y1

                        # --- EXPANSION ---
                        if friendly in GENITALIA_CLASSES:
                            # 300% Radius
                            center_x = x1 + w // 2
                            center_y = y1 + h // 2
                            new_w = int(w * GENITAL_SCALE)
                            new_h = int(h * GENITAL_SCALE)
                            x1 = center_x - new_w // 2
                            y1 = center_y - new_h // 2
                            w, h = new_w, new_h
                        else:
                            # 15% Padding
                            pad_w = int(w * GLOBAL_PADDING)
                            pad_h = int(h * GLOBAL_PADDING)
                            x1 -= pad_w
                            y1 -= pad_h
                            w += (pad_w * 2)
                            h += (pad_h * 2)

                        raw_boxes.append((x1, y1, w, h))

            merged_boxes = merge_boxes_union(raw_boxes)

            matched_indices = set()
            alive_tracks = []

            for track in self.tracks:
                best_iou = 0
                best_idx = -1
                track_rect = track.get_rect()

                for i, new_box in enumerate(merged_boxes):
                    if i in matched_indices: continue
                    iou = calculate_iou(track_rect, new_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_idx != -1 and best_iou > 0.1:
                    track.update(merged_boxes[best_idx])
                    matched_indices.add(best_idx)
                    alive_tracks.append(track)
                else:
                    track.missing_count += 1
                    if track.missing_count < MAX_MISSING_FRAMES:
                        alive_tracks.append(track)

            for i, rect in enumerate(merged_boxes):
                if i not in matched_indices:
                    alive_tracks.append(StickyBox(rect))

            self.tracks = alive_tracks
            
            self.render_list = []
            for track in self.tracks:
                x, y, w, h = track.get_rect()
                
                if w <= 0 or h <= 0: continue
                if x < 0: x = 0
                if y < 0: y = 0
                if x+w > self.w: w = self.w - x
                if y+h > self.h: h = self.h - y

                roi = frame[y:y+h, x:x+w]
                if roi.size == 0: continue 
                
                processed = self.apply_effect(roi, w, h)
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                h_p, w_p, ch = processed.shape
                q_img = QImage(processed.data, w_p, h_p, ch * w_p, QImage.Format_RGB888).copy()
                self.render_list.append((x, y, q_img))
            
            self.update() 
            
        except Exception as e:
            pass

    def paintEvent(self, event):
        painter = QPainter(self)
        for (x, y, q_img) in self.render_list:
            painter.drawImage(x, y, q_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = CensorOverlay()
    sys.exit(app.exec_())