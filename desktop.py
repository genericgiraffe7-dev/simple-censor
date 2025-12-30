import sys
import os
import json
import ctypes
import numpy as np
import cv2
import mss
from ultralytics import YOLO

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QImage

# --- GPU CONFIG ---
DEVICE = '0' 
TARGET_MODEL = "best.onnx"
MODEL_SIZE = 320 
PREFS_FILE = "preferences.json"

CENSOR_CATEGORIES = [
    ("BELLY_EXPOSED", False), ("MALE_GENITALIA_EXPOSED", True), ("BUTTOCKS_EXPOSED", True),
    ("FEMALE_BREAST_EXPOSED", True), ("FEMALE_GENITALIA_EXPOSED", True), ("MALE_BREAST_EXPOSED", False),
    ("ANUS_EXPOSED", True), ("FEET_EXPOSED", False), ("ARMPITS_EXPOSED", False),
    ("FACE_FEMALE", False), ("FACE_MALE", False), ("BELLY_COVERED", False),
    ("FEMALE_GENITALIA_COVERED", False), ("BUTTOCKS_COVERED", False), ("FEET_COVERED", False),
    ("ARMPITS_COVERED", False), ("ANUS_COVERED", False), ("FEMALE_BREAST_COVERED", False)
]

def setup_preferences():
    # --- YOUR NEW REFINED SWEET SPOT ---
    sweet_spot = {
        "global_conf": 0.40,
        "tiled_conf": 0.25,
        "update_delay": 1,
        "smoothing_factor": 0.5,
        "max_missing_frames": 10
    }

    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, 'r') as f:
                prefs = json.load(f)
            print("\n📄 Found existing preferences.")
            if input("Use existing preferences? (y/n): ").lower() == 'y': return prefs
        except: pass

    print("\n--- PERFORMANCE SETUP (James' Updated Recommendations) ---")
    u_delay = int(input(f"   Update Delay (Recommended {sweet_spot['update_delay']}): ") or sweet_spot['update_delay'])
    s_factor = float(input(f"   Smoothing Factor (Recommended {sweet_spot['smoothing_factor']}): ") or sweet_spot['smoothing_factor'])
    m_frames = int(input(f"   Max Missing Frames (Recommended {sweet_spot['max_missing_frames']}): ") or sweet_spot['max_missing_frames'])
    g_conf = float(input(f"   Global Sensitivity (Recommended 40): ") or 40) / 100.0
    t_conf = float(input(f"   Tiled Sensitivity (Recommended 25): ") or 25) / 100.0

    print("\n--- CENSOR CRITERIA ---")
    bl = []
    for cat, default_val in CENSOR_CATEGORIES:
        prompt = "Y/n" if default_val else "y/N"
        choice = input(f" {cat} ({prompt}): ").lower()
        if choice == 'y' or (choice == '' and default_val): bl.append(cat)

    prefs = {"global_conf": g_conf, "tiled_conf": t_conf, "update_delay": u_delay, 
             "smoothing_factor": s_factor, "max_missing_frames": m_frames, "block_list": bl}
    with open(PREFS_FILE, 'w') as f: json.dump(prefs, f, indent=4)
    return prefs

PREFS = setup_preferences()

# --- APP CONFIG ---
INTENSITY_VAL = 45      
GLOBAL_PADDING = 0.22       
GENITAL_SCALE = 3.6         
GENITALIA_CLASSES = ["FEMALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_COVERED", "MALE_GENITALIA_EXPOSED"]

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    ctypes.windll.user32.SetProcessDPIAware()

model = YOLO(os.path.join(os.path.dirname(__file__), "Models", TARGET_MODEL), task='detect')

class MomentumPredictor:
    def __init__(self, rect):
        self.x, self.y, self.w, self.h = rect
        self.vel = np.array([0.0, 0.0])
        self.accel = np.array([0.0, 0.0])
        self.missing_count = 0

    def update(self, new_rect):
        nx, ny, nw, nh = new_rect
        alpha = PREFS["smoothing_factor"]
        
        current_pos = np.array([nx, ny])
        old_pos = np.array([self.x, self.y])
        new_vel = current_pos - old_pos
        
        # Smooth Acceleration
        self.accel = (new_vel - self.vel) * 0.2
        self.vel = new_vel
        
        # Apply Smoothing Factor (0.5)
        self.x = int(alpha * self.x + (1 - alpha) * nx)
        self.y = int(alpha * self.y + (1 - alpha) * ny)
        self.w = int(alpha * self.w + (1 - alpha) * nw)
        self.h = int(alpha * self.h + (1 - alpha) * nh)
        self.missing_count = 0

    def get_rect(self):
        # Apply Lead Prediction
        lead_x = self.x + int(self.vel[0] * 0.5 + self.accel[0])
        lead_y = self.y + int(self.vel[1] * 0.5 + self.accel[1])
        return (lead_x, lead_y, self.w, self.h)

class CensorOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.render_list = []
        self.tracks = []
        screen = QApplication.primaryScreen().geometry()
        self.w, self.h = screen.width(), screen.height()
        self.setGeometry(screen)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)

        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
        except: pass

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        ts = 640 
        self.tiles = [(0,0), (self.w-ts, 0), (self.w//2-ts//2, self.h//2-ts//2), (0, self.h-ts), (self.w-ts, self.h-ts)]
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(PREFS["update_delay"])
        self.show()

    def update_loop(self):
        try:
            raw_screen = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2BGR)
            
            # 1. Global Priority Anchors
            anchors = []
            gf = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE))
            gr = model(gf, conf=PREFS["global_conf"], device=DEVICE, imgsz=MODEL_SIZE, verbose=False)
            gsx, gsy = self.w / MODEL_SIZE, self.h / MODEL_SIZE
            for b in gr[0].boxes:
                name = model.names[int(b.cls[0])]
                if name in PREFS["block_list"]:
                    lx1, ly1, lx2, ly2 = map(int, b.xyxy[0])
                    anchors.append(self.process_box(int(lx1*gsx), int(ly1*gsy), int((lx2-lx1)*gsx), int((ly2-ly1)*gsy), name))

            # 2. Tiled Expansion
            for (tx, ty) in self.tiles:
                crop = frame[ty:ty+640, tx:tx+640]
                tr = model(cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE)), conf=PREFS["tiled_conf"], device=DEVICE, imgsz=MODEL_SIZE, verbose=False)
                ts = 640 / MODEL_SIZE
                for b in tr[0].boxes:
                    name = model.names[int(b.cls[0])]
                    if name in PREFS["block_list"]:
                        lx1, ly1, lx2, ly2 = map(int, b.xyxy[0])
                        t_box = self.process_box(int(lx1*ts)+tx, int(ly1*ts)+ty, int((lx2-lx1)*ts), int((ly2-ly1)*ts), name)
                        
                        matched = False
                        for i in range(len(anchors)):
                            xA, yA = max(t_box[0], anchors[i][0]), max(t_box[1], anchors[i][1])
                            xB, yB = min(t_box[0]+t_box[2], anchors[i][0]+anchors[i][2]), min(t_box[1]+t_box[3], anchors[i][1]+anchors[i][3])
                            inter = max(0, xB-xA) * max(0, yB-yA)
                            iou = inter / float(t_box[2]*t_box[3] + anchors[i][2]*anchors[i][3] - inter + 1e-6)
                            if iou > 0.15:
                                ab = anchors[i]
                                nx, ny = min(ab[0], t_box[0]), min(ab[1], t_box[1])
                                nw = max(ab[0]+ab[2], t_box[0]+t_box[2]) - nx
                                nh = max(ab[1]+ab[3], t_box[1]+t_box[3]) - ny
                                anchors[i] = (nx, ny, nw, nh)
                                matched = True
                                break
                        if not matched: anchors.append(t_box)

            self.sync_and_render(anchors, frame)
        except: pass

    def process_box(self, x, y, w, h, name):
        if name in GENITALIA_CLASSES:
            nw, nh = int(w*GENITAL_SCALE), int(h*GENITAL_SCALE)
            return (x+w//2-nw//2, y+h//2-nh//2, nw, nh)
        pw, ph = int(w*GLOBAL_PADDING), int(h*GLOBAL_PADDING)
        return (x-pw, y-ph, w+pw*2, h+ph*2)

    def sync_and_render(self, final_boxes, frame):
        matched_indices, alive = set(), []
        for track in self.tracks:
            best_iou, best_idx = 0, -1
            tr_rect = track.get_rect()
            for i, nb in enumerate(final_boxes):
                if i in matched_indices: continue
                xA, yA = max(tr_rect[0], nb[0]), max(tr_rect[1], nb[1])
                xB, yB = min(tr_rect[0]+tr_rect[2], nb[0]+nb[2]), min(tr_rect[1]+tr_rect[3], nb[1]+nb[3])
                inter = max(0, xB-xA) * max(0, yB-yA)
                iou = inter / float(tr_rect[2]*tr_rect[3] + nb[2]*nb[3] - inter + 1e-6)
                if iou > best_iou: best_iou, best_idx = iou, i
            
            if best_idx != -1 and best_iou > 0.1:
                track.update(final_boxes[best_idx]); matched_indices.add(best_idx); alive.append(track)
            else:
                track.missing_count += 1
                if track.missing_count < PREFS["max_missing_frames"]: alive.append(track)
        
        for i, rect in enumerate(final_boxes):
            if i not in matched_indices: alive.append(MomentumPredictor(rect))
        self.tracks = alive
        
        self.render_list = []
        for t in self.tracks:
            x, y, w, h = t.get_rect()
            x, y, w, h = max(0, x), max(0, y), min(w, self.w-x), min(h, self.h-y)
            if w > 5 and h > 5:
                roi = frame[y:y+h, x:x+w]
                small = cv2.resize(roi, (max(1, w//INTENSITY_VAL), max(1, h//INTENSITY_VAL)), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                self.render_list.append((x, y, QImage(cv2.cvtColor(pixelated, cv2.COLOR_BGR2RGB).data, w, h, 3*w, QImage.Format_RGB888).copy()))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        for (x, y, q) in self.render_list: p.drawImage(x, y, q)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = CensorOverlay()
    sys.exit(app.exec_())