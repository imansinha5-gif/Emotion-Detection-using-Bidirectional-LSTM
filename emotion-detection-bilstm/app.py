"""Flask backend for the Emotion Detection using BiLSTM website."""
import os
import json
import time
from collections import deque
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from src.preprocessing import preprocess_text

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

model = tokenizer = encoder = None
metrics = {}
load_error = None
history = deque(maxlen=20)

EMOTION_META = {
    "joy": {"emoji": "😊", "color": "#f59e0b", "description": "Happiness, excitement and positive energy."},
    "sadness": {"emoji": "😢", "color": "#3b82f6", "description": "Sadness, disappointment or low mood expressed in text."},
    "anger": {"emoji": "😠", "color": "#ef4444", "description": "Frustration, irritation or anger expressed in text."},
    "fear": {"emoji": "😨", "color": "#8b5cf6", "description": "Worry, fear or uncertainty expressed in text."},
    "love": {"emoji": "❤️", "color": "#ec4899", "description": "Affection, care or love expressed in text."},
    "surprise": {"emoji": "😲", "color": "#14b8a6", "description": "Unexpectedness, amazement or shock expressed in text."},
}

def model_path(filename):
    return os.path.join(MODEL_DIR, filename)

def load_artifacts():
    global model, tokenizer, encoder, metrics, load_error
    required = ["emotion_bilstm.keras", "tokenizer.pkl", "label_encoder.pkl"]
    missing = [name for name in required if not os.path.exists(model_path(name))]
    if missing:
        load_error = (
            "Model files are not trained yet. Run "
            "`python -m src.train --data data/emotion_dataset.csv` first."
        )
        return
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path("emotion_bilstm.keras"))
        tokenizer = joblib.load(model_path("tokenizer.pkl"))
        encoder = joblib.load(model_path("label_encoder.pkl"))
        if os.path.exists(model_path("metrics.json")):
            with open(model_path("metrics.json"), encoding="utf-8") as f:
                metrics = json.load(f)
        load_error = None
    except Exception as exc:
        load_error = f"Could not load model artifacts: {exc}"

def prediction_payload(text):
    if model is None or tokenizer is None or encoder is None:
        raise RuntimeError(load_error or "Model is not available.")
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    clean = preprocess_text(text)
    if not clean:
        raise ValueError("The text contains no usable words after preprocessing.")
    seq = tokenizer.texts_to_sequences([clean])
    x = pad_sequences(seq, maxlen=model.input_shape[1], padding="post", truncating="post")
    probs = model.predict(x, verbose=0)[0]
    labels = list(encoder.classes_)
    scores = {str(label): round(float(prob) * 100, 2) for label, prob in zip(labels, probs)}
    top = str(labels[int(np.argmax(probs))])
    return {
        "emotion": top,
        "confidence": scores[top],
        "probabilities": scores,
        "cleaned_text": clean,
        "meta": EMOTION_META.get(top.lower(), {"emoji": "🙂", "color": "#6366f1", "description": "Predicted emotion."}),
    }

load_artifacts()

@app.route("/")
def index():
    stats = {
        "classes": len(encoder.classes_) if encoder is not None else "—",
        "accuracy": f"{metrics.get('accuracy')}%" if metrics.get("accuracy") is not None else "—",
        "f1": f"{metrics.get('macro_f1')}%" if metrics.get("macro_f1") is not None else "—",
        "size": f"{metrics.get('dataset_size'):,}" if metrics.get("dataset_size") else "—",
        "model": "BiLSTM",
        "status": "Ready" if load_error is None else "Training required",
    }
    return render_template("index.html", error=load_error, stats=stats)

@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if load_error is None else "degraded",
        "model_loaded": model is not None,
        "message": "Emotion detection API is ready." if load_error is None else load_error,
    })

@app.get("/api/info")
def info():
    return jsonify({
        "model": "Bidirectional LSTM",
        "classes": list(map(str, encoder.classes_)) if encoder is not None else [],
        "metrics": metrics,
        "emotion_meta": EMOTION_META,
    })

@app.get("/api/history")
def api_history():
    return jsonify(list(history))

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify(error="Please enter some text."), 400
    if len(text) > 1000:
        return jsonify(error="Text is too long. Maximum 1000 characters."), 400
    try:
        result = prediction_payload(text)
        history.appendleft({
            "text": text[:140],
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "time": time.strftime("%H:%M:%S"),
        })
        return jsonify(result)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify(error=f"Prediction failed: {exc}"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
