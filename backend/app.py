from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path

import os
import re
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer


# =========================
# PATH CONFIG
# =========================

CURRENT_DIR = Path(__file__).resolve().parent

# Works for both:
# 1) Hugging Face Space: app.py and frontend/ are side-by-side
# 2) Local backend folder: app.py is inside backend/, frontend/ is one level above
if (CURRENT_DIR / "frontend").exists():
    BASE_DIR = CURRENT_DIR
else:
    BASE_DIR = CURRENT_DIR.parent

FRONTEND_DIR = BASE_DIR / "frontend"

# Support both possible asset layouts: frontend/assets or root/assets
if (FRONTEND_DIR / "assets").exists():
    ASSETS_DIR = FRONTEND_DIR / "assets"
else:
    ASSETS_DIR = BASE_DIR / "assets"


# =========================
# MODEL CONFIG
# =========================

# Final model: IndoBERT Balance + Augment from Hugging Face model repo.
MODEL_DIR = os.getenv("MODEL_DIR", "littt24/indobert_balance_augment")

# Same as your IndoBERT notebook/PPT configuration.
MAX_LENGTH = 96

# LIME is enabled. Keep this modest because Hugging Face Space CPU is slow.
# Increase to 100+ only for final static screenshots/figures, not live demo.
LIME_NUM_SAMPLES = int(os.getenv("LIME_NUM_SAMPLES", "100"))

TEMPERATURE = 2.0

# Keep 0.5 first so the backend matches the normal binary classifier decision rule.
# If after this fixed app.py the model still over-predicts Clickbait, the next fix is notebook/export.
CLICKBAIT_THRESHOLD = float(os.getenv("CLICKBAIT_THRESHOLD", "0.5"))

# Label mapping used in training/config:
# 0 = non-clickbait, 1 = clickbait
NON_CLICKBAIT_INDEX = 0
CLICKBAIT_INDEX = 1

LABELS = {
    NON_CLICKBAIT_INDEX: "Non-Clickbait",
    CLICKBAIT_INDEX: "Clickbait",
}

CLASS_NAMES = ["Non-Clickbait", "Clickbait"]


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="IndoBERT Clickbait Detection API",
    version="1.0.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST BODY
# =========================

class PredictRequest(BaseModel):
    text: str | None = None
    headline: str | None = None


# =========================
# CLEAN TEXT
# =========================

# IMPORTANT:
# This follows the clean_text() in your IndoBERT notebook:
# remove URL, remove unusual characters, normalize whitespace.
# Do NOT add lowercasing here unless the notebook training also used lowercasing.
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# LOAD MODEL
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CURRENT_DIR:", CURRENT_DIR)
print("BASE_DIR:", BASE_DIR)
print("FRONTEND_DIR:", FRONTEND_DIR)
print("ASSETS_DIR:", ASSETS_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("DEVICE:", device)

local_model_path = Path(MODEL_DIR)

if local_model_path.exists():
    MODEL_SOURCE = str(local_model_path)
    print("Using local model folder:", MODEL_SOURCE)
else:
    MODEL_SOURCE = MODEL_DIR
    print("Using Hugging Face model repo:", MODEL_SOURCE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)

# Force clean binary label metadata at runtime.
# This does not change model weights; it only prevents confusing config metadata.
model.config.num_labels = 2
model.config.id2label = {
    0: "non-clickbait",
    1: "clickbait",
}
model.config.label2id = {
    "non-clickbait": 0,
    "clickbait": 1,
}

model.to(device)
model.eval()

print("Model loaded successfully.")
print("model.config.num_labels:", model.config.num_labels)
print("model.config.id2label:", model.config.id2label)
print("model.config.label2id:", model.config.label2id)
print("LABEL MAPPING: 0 = Non-Clickbait, 1 = Clickbait")


# =========================
# PREDICTION FUNCTION
# =========================

def predict_proba(texts):
    cleaned_texts = [clean_text(text) for text in texts]

    inputs = tokenizer(
        cleaned_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits / TEMPERATURE
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.cpu().numpy()


def predict_single_text(text: str):
    cleaned_text = clean_text(text)
    probabilities = predict_proba([cleaned_text])[0]

    non_clickbait_prob = float(probabilities[NON_CLICKBAIT_INDEX])
    clickbait_prob = float(probabilities[CLICKBAIT_INDEX])

    # Decision is based specifically on the Clickbait probability.
    # This avoids the old bug where max confidence could be mistaken as clickbait score.
    is_clickbait = clickbait_prob >= CLICKBAIT_THRESHOLD

    if is_clickbait:
        predicted_class = CLICKBAIT_INDEX
        confidence = clickbait_prob
    else:
        predicted_class = NON_CLICKBAIT_INDEX
        confidence = non_clickbait_prob

    return cleaned_text, probabilities, predicted_class, confidence


# =========================
# LIME EXPLANATION
# =========================

lime_explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    random_state=42,
)


def generate_lime_explanation(text: str, predicted_class: int):
    try:
        cleaned_text = clean_text(text)

        explanation = lime_explainer.explain_instance(
            text_instance=cleaned_text,
            classifier_fn=predict_proba,
            labels=[predicted_class],
            num_features=8,
            num_samples=LIME_NUM_SAMPLES,
        )

        lime_list = explanation.as_list(label=predicted_class)
        result = []

        for word, weight in lime_list:
            weight = float(weight)
            result.append({
                "word": word,
                "weight": round(weight, 6),
                "abs_weight": round(abs(weight), 6),
                "impact": "supports_prediction" if weight > 0 else "against_prediction",
            })

        return result

    except Exception as e:
        print("LIME ERROR:", e)
        return []


# =========================
# API ROUTES
# =========================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_dir": MODEL_DIR,
        "model_source": MODEL_SOURCE,
        "device": str(device),
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "threshold": CLICKBAIT_THRESHOLD,
        "lime_num_samples": LIME_NUM_SAMPLES,
        "label_mapping": {
            "0": "Non-Clickbait",
            "1": "Clickbait",
        },
        "id2label_from_model": model.config.id2label,
        "label2id_from_model": model.config.label2id,
    }


@app.get("/debug")
def debug():
    model_files = []

    if local_model_path.exists():
        model_files = sorted([p.name for p in local_model_path.iterdir()])

    return {
        "current_dir": str(CURRENT_DIR),
        "base_dir": str(BASE_DIR),
        "frontend_dir": str(FRONTEND_DIR),
        "assets_dir": str(ASSETS_DIR),
        "model_dir": MODEL_DIR,
        "model_source": MODEL_SOURCE,
        "model_files": model_files,
        "device": str(device),
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "clickbait_threshold": CLICKBAIT_THRESHOLD,
        "lime_num_samples": LIME_NUM_SAMPLES,
        "non_clickbait_index": NON_CLICKBAIT_INDEX,
        "clickbait_index": CLICKBAIT_INDEX,
        "id2label_from_model": model.config.id2label,
        "label2id_from_model": model.config.label2id,
    }


def build_prediction_response(text: str):
    cleaned_text, probabilities, predicted_class, confidence = predict_single_text(text)

    non_clickbait_prob = float(probabilities[NON_CLICKBAIT_INDEX])
    clickbait_prob = float(probabilities[CLICKBAIT_INDEX])

    prediction_label = LABELS[predicted_class]
    is_clickbait = bool(predicted_class == CLICKBAIT_INDEX)

    lime_explanation = generate_lime_explanation(
        text=cleaned_text,
        predicted_class=predicted_class,
    )

    return {
        "input_text": text,
        "headline": text,
        "cleaned_text": cleaned_text,
        "cleaned_headline": cleaned_text,

        "prediction_id": predicted_class,
        "prediction_label": prediction_label,
        "prediction": prediction_label,
        "label": prediction_label,
        "isClickbait": is_clickbait,

        # Confidence = probability of the predicted class.
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),

        # Separate scores for debugging/frontend.
        "clickbait_score": round(clickbait_prob, 4),
        "clickbait_score_percent": round(clickbait_prob * 100, 2),
        "non_clickbait_score": round(non_clickbait_prob, 4),
        "non_clickbait_score_percent": round(non_clickbait_prob * 100, 2),

        # Frontend compatibility format 1.
        "probabilities": {
            "Non-Clickbait": round(non_clickbait_prob, 4),
            "Clickbait": round(clickbait_prob, 4),
        },

        # Frontend compatibility format 2.
        "probability": {
            "non_clickbait": round(non_clickbait_prob, 4),
            "clickbait": round(clickbait_prob, 4),
        },

        "lime_explanation": lime_explanation,
        "lime": lime_explanation,

        "temperature": TEMPERATURE,
        "threshold": CLICKBAIT_THRESHOLD,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    text = (request.text or request.headline or "").strip()

    if text == "":
        return JSONResponse(
            status_code=400,
            content={"error": "Text/headline cannot be empty."},
        )

    return build_prediction_response(text)


@app.get("/predict")
def predict_get(text: str):
    return build_prediction_response(text)


@app.get("/predict_raw")
def predict_raw(text: str):
    """Fast debug endpoint without LIME."""
    cleaned_text = clean_text(text)
    probabilities = predict_proba([cleaned_text])[0]

    return {
        "input_text": text,
        "cleaned_text": cleaned_text,
        "raw_prob_index_0_non_clickbait": round(float(probabilities[0]), 6),
        "raw_prob_index_1_clickbait": round(float(probabilities[1]), 6),
        "threshold": CLICKBAIT_THRESHOLD,
        "label_mapping": {
            "0": "Non-Clickbait",
            "1": "Clickbait",
        },
    }


# =========================
# FRONTEND ROUTES
# =========================

@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    return {
        "message": "IndoBERT Clickbait Detection API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/index.html")
def serve_index_html():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/tentang")
def serve_tentang():
    return FileResponse(FRONTEND_DIR / "tentang.html")


@app.get("/tentang.html")
def serve_tentang_html():
    return FileResponse(FRONTEND_DIR / "tentang.html")


@app.get("/style.css")
def serve_style():
    return FileResponse(FRONTEND_DIR / "style.css")


@app.get("/script.js")
def serve_script():
    return FileResponse(FRONTEND_DIR / "script.js")


if FRONTEND_DIR.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=FRONTEND_DIR),
        name="frontend",
    )

if ASSETS_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=ASSETS_DIR),
        name="assets",
    )