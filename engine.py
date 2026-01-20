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
from pytorch_grad_cam.utils.image import show_cam_on_image

DB_PATH = "cropvision_history.db"
EFFNET_PATH = "cropvision_effnet.pth"
UNET_PATH = "cropvision_unet.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_models():
    global effnet_model, unet_model, gradcam_cam

    if effnet_model is None:
        effnet_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=3).to(device)
        effnet_model.eval()
        gradcam_cam = GradCAM(model=effnet_model, target_layers=[effnet_model.conv_head])

    if unet_model is None:
        unet_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1).to(device)
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

def analyze_crop_leaf(image_bytes, filename="leaf_image.jpg"):
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
            "reason": f"Low quality or non-leaf image. Sharpness: {blur_score:.1f}, Leaf: {leaf_ratio:.1f}%"
        }

    input_tensor = inference_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_clf(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    classes = ["Angular Leaf Spot", "Bean Rust", "Healthy Leaf"]
    disease_label = classes[pred_idx] if pred_idx < len(classes) else "Unknown"
    confidence = round(float(probs[pred_idx] * 100.0), 1)

    rgb_224 = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
    heatmap = cam(input_tensor=input_tensor, targets=None)[0, :]
    gradcam_img = show_cam_on_image(rgb_224, heatmap, use_rgb=True)

    return {
        "valid": True,
        "disease": disease_label,
        "confidence": confidence,
        "gradcam_img": encode_image(gradcam_img)
    }

def get_prediction_history(limit=15):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, filename, disease, confidence, severity, severity_stage, status FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
        return cursor.fetchall()
