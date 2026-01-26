import io
import cv2
import base64
import sqlite3
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import segmentation_models_pytorch as smp
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

DB_PATH = "cropvision_history.db"
EFFNET_PATH = "cropvision_effnet_best.pth"
UNET_PATH = "cropvision_unet_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_disease_name(raw_name):
    if "___" in raw_name:
        crop_part, disease_part = raw_name.split("___", 1)
    else:
        crop_part, disease_part = "", raw_name

    crop = crop_part.replace(",", "").replace("_", " ").strip()
    disease = disease_part.replace("_", " ").strip()

    c_words = [w.capitalize() if not (w.startswith("(") and w.endswith(")")) else f"({w[1:-1].capitalize()})" for w in crop.split()]
    d_words = [w.capitalize() for w in disease.split()]

    c_str = " ".join(c_words)
    d_str = " ".join(d_words)

    if d_str.lower() == "healthy":
        d_str = "Healthy Leaf"

    return f"{c_str} - {d_str}" if c_str else d_str

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                disease TEXT,
                confidence REAL,
                uncertainty REAL,
                severity REAL,
                severity_stage TEXT,
                status TEXT
            )
        """)
        conn.commit()

init_db()

effnet_model = None
unet_model = None
gradcam_cam = None
class_names = []
clean_class_names = []

def load_models():
    global effnet_model, unet_model, gradcam_cam, class_names, clean_class_names

    if effnet_model is None:
        checkpoint = torch.load(EFFNET_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            weights = checkpoint["model_state_dict"]
            class_names = checkpoint.get("class_names", [])
        else:
            weights = checkpoint
            class_names = []

        num_classes = len(class_names) if class_names else 38
        clean_class_names = [format_disease_name(c) for c in class_names]

        effnet_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes).to(device)
        effnet_model.load_state_dict(weights)
        effnet_model.eval()

        gradcam_cam = GradCAM(model=effnet_model, target_layers=[effnet_model.conv_head])

    if unet_model is None:
        unet_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
        unet_model.load_state_dict(torch.load(UNET_PATH, map_location=device))
        unet_model.eval()

    return effnet_model, unet_model, gradcam_cam

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def encode_image(img_arr):
    pil_img = Image.fromarray(img_arr.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def analyze_crop_leaf(image_bytes, filename="leaf_image.jpg", crop_filter="Auto-Detect"):
    model_clf, model_seg, cam = load_models()

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb_img = np.array(pil_img.resize((256, 256)))
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    leaf_mask = cv2.inRange(hsv, np.array([10, 25, 20]), np.array([95, 255, 255]))
    leaf_ratio = float((np.sum(leaf_mask > 0) / leaf_mask.size) * 100)

    if blur_score < 25.0 or leaf_ratio < 5.0:
        return {
            "valid": False,
            "reason": f"Low quality or non-leaf image. Sharpness Score: {blur_score:.1f} (Min: 25), Leaf Ratio: {leaf_ratio:.1f}% (Min: 5%)"
        }

    input_tensor = inference_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_clf(input_tensor)[0]

    # Crop Domain Filtering (allows user to lock to specific crop e.g. Apple, Tomato)
    active_indices = list(range(len(class_names)))
    if crop_filter and crop_filter != "Auto-Detect":
        filter_key = crop_filter.split()[0].lower()
        matched = [i for i, c in enumerate(class_names) if filter_key in c.lower()]
        if matched:
            active_indices = matched

    sub_logits = logits[active_indices]
    sub_probs = F.softmax(sub_logits, dim=0).cpu().numpy()

    best_sub_idx = int(np.argmax(sub_probs))
    pred_idx = active_indices[best_sub_idx]
    confidence = round(float(sub_probs[best_sub_idx] * 100.0), 1)
    uncertainty = round(float(max(1.0, 100.0 - confidence)), 1)

    raw_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
    disease_label = clean_class_names[pred_idx] if pred_idx < len(clean_class_names) else raw_label
    is_healthy = "healthy" in raw_label.lower()

    top_sub_order = np.argsort(sub_probs)[::-1][:min(3, len(sub_probs))]
    top_candidates = [
        {
            "disease": clean_class_names[active_indices[i]] if active_indices[i] < len(clean_class_names) else class_names[active_indices[i]],
            "confidence": round(float(sub_probs[i] * 100.0), 1)
        }
        for i in top_sub_order
    ]

    if confidence < 45.0:
        status = "Uncertain Prediction"
    elif is_healthy:
        status = "Verified Healthy Crop"
    else:
        status = "Verified Prediction"

    # Target-Guided Grad-CAM Attention Map
    rgb_224 = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
    targets = [ClassifierOutputTarget(pred_idx)]
    raw_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Restrict attention heatmap strictly to the leaf boundary to avoid background bleeding
    leaf_mask_224 = cv2.resize(leaf_mask, (224, 224)) > 0
    bounded_cam = raw_cam.copy()
    bounded_cam[~leaf_mask_224] = 0.0

    cam_max = bounded_cam.max()
    cam_min = bounded_cam[leaf_mask_224].min() if np.any(leaf_mask_224) else 0.0
    if cam_max > cam_min:
        bounded_cam[leaf_mask_224] = (bounded_cam[leaf_mask_224] - cam_min) / (cam_max - cam_min + 1e-7)

    gradcam_img = show_cam_on_image(rgb_224, bounded_cam, use_rgb=True)

    # U-Net Lesion Feature Extraction & Hybrid Pathology Segmentation
    norm_rgb = (rgb_img / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    seg_input = torch.tensor(norm_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_prob = torch.sigmoid(model_seg(seg_input))[0, 0].cpu().numpy()

    # Healthy green tissue mask
    green_mask = cv2.inRange(hsv, np.array([30, 35, 30]), np.array([95, 255, 255]))
    total_leaf_pixels = max(int(np.sum(leaf_mask > 0)), 1)

    if is_healthy:
        lesion_mask = np.zeros_like(leaf_mask)
        severity_pct = 0.0
        severity_stage = "Healthy"
    else:
        color_lesion = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(green_mask))
        unet_active = (seg_prob > 0.50).astype(np.uint8) * 255
        
        raw_lesion = cv2.bitwise_and(color_lesion, unet_active)
        if np.sum(raw_lesion > 0) < 50:
            raw_lesion = color_lesion

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lesion_mask = cv2.morphologyEx(raw_lesion, cv2.MORPH_OPEN, kernel)
        lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)

        infected_pixels = int(np.sum(lesion_mask > 0))
        severity_pct = float((infected_pixels / total_leaf_pixels) * 100.0)

        if severity_pct < 5.0:
            severity_stage = "Mild"
        elif severity_pct < 20.0:
            severity_stage = "Moderate"
        else:
            severity_stage = "Severe"

    boxes_img = rgb_img.copy()
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_count = 0
    if not is_healthy:
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(boxes_img, (x, y), (x + w, y + h), (239, 68, 68), 2)
                spot_count += 1

    overlay = rgb_img.copy()
    overlay[lesion_mask > 0] = [239, 68, 68]
    blended_seg = cv2.addWeighted(rgb_img, 0.65, overlay, 0.35, 0)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (timestamp, filename, disease, confidence, uncertainty, severity, severity_stage, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            disease_label,
            round(confidence, 1),
            round(uncertainty, 1),
            round(severity_pct, 1),
            severity_stage,
            status
        ))
        conn.commit()

    return {
        "valid": True,
        "disease": disease_label,
        "confidence": round(confidence, 1),
        "uncertainty": round(uncertainty, 1),
        "severity_pct": round(severity_pct, 1),
        "severity_stage": severity_stage,
        "spots_detected": spot_count,
        "blur_score": round(blur_score, 1),
        "status": status,
        "top_predictions": top_candidates,
        "yolo_img": encode_image(boxes_img),
        "gradcam_img": encode_image(gradcam_img),
        "seg_img": encode_image(blended_seg)
    }

def get_prediction_history(limit=15):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, filename, disease, confidence, severity, severity_stage, status 
            FROM predictions 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
