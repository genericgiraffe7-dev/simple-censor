import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import gradio as gr
import cv2
import os
import numpy as np
from ultralytics import YOLO
from moviepy import VideoFileClip

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

custom_css = ".resizable-container { resize: vertical; overflow: auto; min-height: 400px; border: 3px solid #4f46e5; border-radius: 8px; padding: 10px; }"
js_shortcuts = "function(x) { document.addEventListener('keydown', (e) => { if (e.key === 'ArrowRight') { document.getElementById('btn-next').click(); } else if (e.key === 'ArrowLeft') { document.getElementById('btn-prev').click(); } }); return x; }"

LABEL_MAP = {
    "MALE_GENITALIA_EXPOSED": "Genitalia (Male Exposed)",
    "FEMALE_GENITALIA_EXPOSED": "Genitalia (Female Exposed)",
    "FEMALE_BREAST_EXPOSED": "Breasts (Exposed)",
    "BUTTOCKS_EXPOSED": "Buttocks (Exposed)",
    "ANUS_EXPOSED": "Anus (Exposed)",
    "BELLY_EXPOSED": "Belly (Exposed)",
    "MALE_BREAST_EXPOSED": "Chest (Male)",
    "FEET_EXPOSED": "Feet (Exposed)",
    "ARMPITS_EXPOSED": "Armpits (Exposed)",
    "FACE_FEMALE": "Face (Female)",
    "FACE_MALE": "Face (Male)",
    "BELLY_COVERED": "Belly (Covered)",
    "FEMALE_GENITALIA_COVERED": "Genitalia (Female Covered)",
    "BUTTOCKS_COVERED": "Buttocks (Covered)",
    "FEET_COVERED": "Feet (Covered)",
    "ARMPITS_COVERED": "Armpits (Covered)",
    "ANUS_COVERED": "Anus (Covered)",
    "FEMALE_BREAST_COVERED": "Breasts (Covered)",
    "person": "Full Body",
    "face": "General Face",
    "closed": "Eye (Closed)",
    "open": "Eye (Open)"
}

DEFAULT_CENSOR = ["Genitalia (Female Exposed)", "Genitalia (Male Exposed)", "Breasts (Exposed)", "Buttocks (Exposed)", "Anus (Exposed)"]

# --- MODEL LOADER ---
def load_all_models():
    if not os.path.exists(MODELS_DIR): return []
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.onnx')]
    loaded = []
    
    import onnxruntime as ort
    os.environ["ORT_LOGGING_LEVEL"] = "3" 

    for f in files:
        try:
            task = 'segment' if "seg" in f.lower() else 'detect'
            m = YOLO(os.path.join(MODELS_DIR, f), task=task)
            loaded.append(m)
        except: pass
    return loaded

models_list = load_all_models()

# --- PROCESSING ENGINE ---
def apply_custom_effect(img, x1, y1, x2, y2, style, intensity, color_hex):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return img
    h, w = roi.shape[:2]
    if "Pixelate" in style:
        b = max(1, int(intensity / 2))
        small = cv2.resize(roi, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif "Blur" in style:
        k = max(1, int(intensity * 2) + 1)
        k = k if k % 2 != 0 else k+1
        roi = cv2.GaussianBlur(roi, (k, k), 0)
    elif "Solid" in style:
        color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        overlay = roi.copy(); overlay[:] = color
        alpha = intensity / 100.0
        roi = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    img[y1:y2, x1:x2] = roi
    return img

def process_frame(frame, censor_rules, keep_rules, style, color, intensity, expansion, visor):
    h, w = frame.shape[:2]
    all_faces, censor_eyes, other_items, censor_masks, to_keep = [], [], [], [], []
    
    for model in models_list:
        try:
            imgsz = 640 if model.overrides.get('task') == 'segment' else 320
            results = model(frame, imgsz=imgsz, verbose=False)
        except: continue
        for res in results:
            if res.boxes:
                for i, box in enumerate(res.boxes):
                    raw_name = model.names.get(int(box.cls[0]), "unknown")
                    friendly = LABEL_MAP.get(raw_name, raw_name)
                    coords = [int(c) for c in box.xyxy[0]]
                    if "Face" in friendly or "face" in raw_name.lower(): all_faces.append(coords)
                    if friendly in keep_rules: to_keep.append(coords)
                    elif friendly in censor_rules:
                        if hasattr(res, 'masks') and res.masks is not None:
                            m_resized = cv2.resize(res.masks.data[i].cpu().numpy(), (w, h))
                            mask_uint8 = (m_resized > 0.5).astype(np.uint8)
                            if expansion != 0:
                                k = abs(int(expansion))
                                kernel = np.ones((k, k), np.uint8)
                                mask_uint8 = cv2.dilate(mask_uint8, kernel) if expansion > 0 else cv2.erode(mask_uint8, kernel)
                            censor_masks.append(mask_uint8 > 0)
                        else:
                            if "Eye" in friendly: censor_eyes.append(coords)
                            else: other_items.append(coords)
                            
    eff_layer = apply_custom_effect(frame.copy(), 0, 0, w, h, style, intensity, color)
    m_mask = np.zeros((h, w), dtype=np.uint8)
    for m in censor_masks: m_mask[m] = 255
    
    eyes_used = [False] * len(censor_eyes)
    if visor and all_faces:
        for f in all_faces:
            face_eyes = [censor_eyes[idx] for idx, e in enumerate(censor_eyes) if not eyes_used[idx] and not (e[0] > f[2] or e[2] < f[0] or e[1] > f[3] or e[3] < f[1])]
            for idx, e in enumerate(censor_eyes): 
                if e in face_eyes: eyes_used[idx] = True
            if face_eyes:
                avg_y = sum([(e[1]+e[3])//2 for e in face_eyes]) // len(face_eyes)
                thick = max(5, int((f[3]-f[1]) * 0.18 + expansion))
                cv2.line(m_mask, (f[0], avg_y), (f[2], avg_y), 255, thick)

    for idx, e in enumerate(censor_eyes):
        if not eyes_used[idx]: cv2.rectangle(m_mask, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), 255, -1)
    for (x1, y1, x2, y2) in other_items:
        ex = int(expansion)
        rx1, ry1, rx2, ry2 = max(0, x1-ex), max(0, y1-ex), min(w, x2+ex), min(h, y2+ex)
        if rx2 > rx1 and ry2 > ry1: cv2.rectangle(m_mask, (rx1, ry1), (rx2, ry2), 255, -1)
    for (x1, y1, x2, y2) in to_keep: cv2.rectangle(m_mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
    frame[m_mask > 0] = eff_layer[m_mask > 0]
    return frame

# --- FILE HANDLER ---
def process_single_file(file_path, sensitivity, censor_rules, keep_rules, style, color, intensity, expansion, visor):
    try:
        conf = sensitivity / 100.0
        name, ext = os.path.splitext(os.path.basename(file_path))
        is_video = ext.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif']
        
        if is_video:
            out_p = os.path.join(OUTPUT_FOLDER, f"processed_{name}.mp4")
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(3)), int(cap.get(4))
            tmp = os.path.join(OUTPUT_FOLDER, f"temp_{name}.mp4")
            out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                for m in models_list: m.conf = conf
                out.write(process_frame(frame, censor_rules, keep_rules, style, color, intensity, expansion, visor))
            cap.release(); out.release()

            try:
                orig_clip = VideoFileClip(file_path)
                proc_clip = VideoFileClip(tmp)
                final_clip = proc_clip.with_audio(orig_clip.audio) if orig_clip.audio else proc_clip
                final_clip.write_videofile(out_p, codec="libx264", audio_codec="aac", logger=None)
                orig_clip.close(); proc_clip.close(); final_clip.close()
                if os.path.exists(tmp): os.remove(tmp)
                return out_p
            except: return tmp
        else:
            out_p = os.path.join(OUTPUT_FOLDER, f"processed_{name}{ext}")
            frame = cv2.imread(file_path)
            for m in models_list: m.conf = conf
            cv2.imwrite(out_p, process_frame(frame, censor_rules, keep_rules, style, color, intensity, expansion, visor))
            return out_p
    except: return file_path

def batch_processor(file_list, sensitivity, censor_rules, keep_rules, style, color, intensity, expansion, visor):
    if not file_list: return None, "No files uploaded."
    res = [process_single_file(f.name, sensitivity, censor_rules, keep_rules, style, color, intensity, expansion, visor) for f in file_list]
    return res, f"Processed {len(res)} files."

# --- UI INTERFACE ---
with gr.Blocks(title="SimpleCensor") as app:
    st_files, st_idx = gr.State([]), gr.State(0)
    gr.Markdown("# SimpleCensor V3.0")
    with gr.Tabs():
        with gr.TabItem("Batch Process"):
            with gr.Row():
                with gr.Column(scale=1):
                    in_f = gr.File(label="Upload", file_count="multiple")
                    btn_p = gr.Button("Process", variant="primary")
                    stat = gr.Textbox(label="Status", interactive=False)
                    s_drp = gr.Dropdown(["Pixelate", "Blur", "Solid Color"], value="Pixelate", label="Style")
                    i_sld = gr.Slider(0, 100, 50, label="Strength")
                    e_sld = gr.Slider(-50, 50, 10, label="Expansion")
                    c_pk = gr.ColorPicker(label="Color", value="#000000")
                    sens = gr.Slider(1, 100, 20, label="Sensitivity Percent")
                    visor = gr.Checkbox(True, label="Visor Mode")
                    c_sel = gr.CheckboxGroup(sorted(list(LABEL_MAP.values())), value=DEFAULT_CENSOR, label="Censor Filters")
                    k_sel = gr.CheckboxGroup(sorted(list(LABEL_MAP.values())), label="Exceptions")
                with gr.Column(scale=2):
                    with gr.Group():
                        m_img = gr.Image(label="Preview Image", visible=False)
                        m_vid = gr.Video(label="Preview Video", visible=False)
                    with gr.Row():
                        b_prv, b_nxt = gr.Button("Previous"), gr.Button("Next")
        with gr.TabItem("Manual Refine"):
            with gr.Row():
                m_editor = gr.ImageEditor(label="Editor", type="numpy")
                m_result = gr.Image(label="Result")
            btn_m = gr.Button("Apply Manual Effect")

    def update_v(f, i):
        if not f: return 0, gr.update(visible=False), gr.update(visible=False)
        idx = i % len(f)
        path = f[idx]
        is_v = path.lower().endswith(('.mp4', '.mov', '.avi', '.webm'))
        return idx, gr.update(value=path if not is_v else None, visible=not is_v), gr.update(value=path if is_v else None, visible=is_v)

    def on_p(f, s, c, k, st, cp, i, e, v):
        res, msg = batch_processor(f, s, c, k, st, cp, i, e, v)
        if not res: return [], msg, 0, gr.update(visible=False), gr.update(visible=False)
        new_idx, img_up, vid_up = update_v(res, 0)
        return res, msg, new_idx, img_up, vid_up

    btn_p.click(on_p, [in_f, sens, c_sel, k_sel, s_drp, c_pk, i_sld, e_sld, visor], [st_files, stat, st_idx, m_img, m_vid])
    b_prv.click(lambda f, i: update_v(f, i - 1), [st_files, st_idx], [st_idx, m_img, m_vid])
    b_nxt.click(lambda f, i: update_v(f, i + 1), [st_files, st_idx], [st_idx, m_img, m_vid])

    def manual_refine_logic(editor_data, style, color, intensity):
        if not editor_data or editor_data["background"] is None: return None
        img = editor_data["background"].copy()
        if editor_data["layers"]:
            for layer in editor_data["layers"]:
                mask = cv2.cvtColor(layer, cv2.COLOR_RGBA2GRAY)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    img = apply_custom_effect(img, int(x), int(y), int(x+w), int(y+h), style, intensity, color)
        return img
    btn_m.click(manual_refine_logic, [m_editor, s_drp, c_pk, i_sld], m_result)

app.launch(inbrowser=True, allowed_paths=[OUTPUT_FOLDER], theme=gr.themes.Soft(), css=custom_css, js=js_shortcuts)