from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import base64
import tempfile

import torch
from PIL import Image
from torchvision import transforms
from transformers import EfficientNetForImageClassification

import cv2
import numpy as np

app = Flask(__name__)

ALLOWED_ORIGINS = [
    "https://fracture-frontend-six.vercel.app",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "null"
]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")

NUM_LABELS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 这里替换成你真实姓名和学号
AUTHOR_NAME = "你的姓名"
AUTHOR_ID = "你的学号"

CLASSES = [
    "正常",
    "轻微骨折",
    "裂纹骨折",
    "移位骨折",
    "粉碎性骨折",
    "关节附近骨折",
    "愈合期骨折",
    "骨质异常",
    "高风险异常影像",
    "其他异常"
]

CLASS_EXPLANATIONS = {
    "正常": "未见明显骨折征象，建议结合临床表现进一步判断。",
    "轻微骨折": "疑似轻度骨折或细小断裂线，需要进一步影像复核。",
    "裂纹骨折": "疑似裂纹样骨折，通常表现为细线状异常影像。",
    "移位骨折": "疑似骨折并伴随位置偏移，风险较高。",
    "粉碎性骨折": "疑似多段破裂样骨折，需要重点关注。",
    "关节附近骨折": "疑似关节周围骨折，可能影响活动功能。",
    "愈合期骨折": "疑似骨折修复期影像表现。",
    "骨质异常": "存在骨质结构异常，建议进一步检查。",
    "高风险异常影像": "图像存在明显异常特征，建议尽快复核。",
    "其他异常": "检测到异常，但未能明确归入主要类别。"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = None


def load_model():
    model = EfficientNetForImageClassification.from_pretrained(
        "google/efficientnet-b0",
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        model = checkpoint

    model.to(DEVICE)
    model.eval()
    return model


def get_model():
    global model
    if model is None:
        model = load_model()
    return model


def image_from_pil(pil_image):
    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    return image_tensor


def classify_pil_image(pil_image):
    model = get_model()
    image_tensor = image_from_pil(pil_image)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    prediction = CLASSES[pred_idx]
    is_fracture = pred_idx != 0

    if pred_idx == 0:
        severity = "无明显骨折"
        diagnosis = "未检测到明显骨折"
    elif pred_idx in [1, 2]:
        severity = "轻度"
        diagnosis = "存在轻度骨折风险"
    elif pred_idx in [3, 5]:
        severity = "中度"
        diagnosis = "存在中度骨折风险"
    elif pred_idx == 4:
        severity = "重度"
        diagnosis = "存在重度骨折风险"
    else:
        severity = "待复核"
        diagnosis = "存在异常影像风险"

    explanation = CLASS_EXPLANATIONS.get(prediction, "模型检测到异常特征，建议复核。")

    probabilities = [
        {
            "label": CLASSES[i],
            "probability": round(float(probs[i]), 4)
        }
        for i in range(len(CLASSES))
    ]
    probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "prediction": prediction,
        "class_id": pred_idx,
        "confidence": round(confidence, 4),
        "diagnosis": diagnosis,
        "is_fracture": is_fracture,
        "severity": severity,
        "explanation": explanation,
        "top_predictions": probabilities[:5]
    }


def pil_from_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Fracture Monitoring API is running",
        "author_name": AUTHOR_NAME,
        "author_id": AUTHOR_ID
    })


@app.get("/health")
def health():
    try:
        get_model()
        return jsonify({"status": "ok", "message": "model loaded successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(file).convert("RGB")
        result = classify_pil_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/predict_frame")
def predict_frame():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data"}), 400

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = classify_pil_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/predict_video")
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        sample_interval = max(int(fps), 1)  # 每秒取1帧
        frame_count = 0
        sampled_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0:
                pil_image = pil_from_frame(frame)
                result = classify_pil_image(pil_image)
                sampled_results.append(result)

            frame_count += 1

        cap.release()

        if not sampled_results:
            return jsonify({"error": "No frames analyzed"}), 400

        fracture_votes = sum(1 for r in sampled_results if r["is_fracture"])
        normal_votes = len(sampled_results) - fracture_votes

        severity_count = {}
        class_count = {}
        avg_conf = 0.0

        for r in sampled_results:
            severity_count[r["severity"]] = severity_count.get(r["severity"], 0) + 1
            class_count[r["prediction"]] = class_count.get(r["prediction"], 0) + 1
            avg_conf += r["confidence"]

        avg_conf /= len(sampled_results)
        final_prediction = max(class_count, key=class_count.get)
        final_severity = max(severity_count, key=severity_count.get)
        final_is_fracture = fracture_votes > normal_votes
        final_diagnosis = "视频中存在骨折风险" if final_is_fracture else "视频中未检测到明显骨折"

        return jsonify({
            "prediction": final_prediction,
            "diagnosis": final_diagnosis,
            "is_fracture": final_is_fracture,
            "severity": final_severity,
            "confidence": round(avg_conf, 4),
            "frames_analyzed": len(sampled_results),
            "fracture_votes": fracture_votes,
            "normal_votes": normal_votes,
            "top_summary": class_count
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("🚀 Fracture Monitoring API starting...")
    app.run(host="0.0.0.0", port=port, debug=False)