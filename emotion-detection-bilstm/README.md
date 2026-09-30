# EmotionAI — Emotion Detection Using Bidirectional LSTM

A complete B.Tech CSE project that combines NLP preprocessing, a TensorFlow/Keras Bidirectional LSTM model, a Flask REST backend, and a responsive HTML/CSS/JavaScript frontend.

## Features

- Six-class emotion prediction: joy, sadness, anger, fear, love, surprise
- BiLSTM text-classification model
- Shared preprocessing between training and inference
- Probability distribution and confidence display
- Responsive dark glassmorphism UI
- Local emotion gallery — no external image assets required
- Flask JSON prediction API
- Health and model-info endpoints
- In-memory recent prediction history
- Sample dataset included
- Render deployment configuration
- GitHub-ready project structure

## Architecture

`Text → Cleaning → Tokenization → Padding → Embedding → BiLSTM → Dropout → Dense → Softmax → Emotion`

## Project structure

```text
emotion-detection-bilstm/
├── app.py
├── requirements.txt
├── Procfile
├── render.yaml
├── data/
│   └── emotion_dataset.csv
├── models/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── train.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run locally on Windows

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train --data data/emotion_dataset.csv --epochs 15 --batch 32
python app.py
```

Open `http://127.0.0.1:5000`.

### Faster demo training

The included dataset is intentionally small for a college demo. For a quick test:

```powershell
python -m src.train --data data/emotion_dataset.csv --epochs 3 --batch 16
```

For meaningful model evaluation, replace the sample CSV with a larger, properly licensed emotion dataset.

## API

### `GET /health`
Returns whether the model is loaded.

### `GET /api/info`
Returns model metadata, classes and saved metrics.

### `GET /api/history`
Returns the last 20 predictions held in memory.

### `POST /predict`

JSON body:

```json
{"text":"I am really excited about my new project!"}
```

Example response:

```json
{
  "emotion":"joy",
  "confidence":91.4,
  "probabilities":{
    "anger":0.3,
    "fear":0.7,
    "joy":91.4,
    "love":4.8,
    "sadness":1.1,
    "surprise":1.7
  }
}
```

## Deployment on Render

This repository includes `render.yaml` and `Procfile`.

Render build:
```text
pip install -r requirements.txt && python -m src.train --data data/emotion_dataset.csv --epochs 3 --batch 16
```

Render start:
```text
gunicorn app:app
```

For production, use a substantially larger training dataset and consider storing model artifacts outside Git or using Git LFS. TensorFlow builds can require significant build time and memory.

## Important note about the included model

The repository contains the training pipeline, not a pre-trained model binary. Run the training command once to generate the model artifacts. Do not claim an accuracy number until the model has actually been trained and evaluated.

## Academic concepts

- NLP preprocessing
- Tokenization and padding
- Word embeddings
- LSTM and Bidirectional LSTM
- Dropout and regularization
- Softmax multi-class classification
- Accuracy, precision, recall and F1-score
- Confusion matrix
- REST API integration
- Full-stack ML deployment

## Limitations

Sarcasm, slang, spelling mistakes, code-mixed language and ambiguous context can cause incorrect predictions. The model only predicts categories represented in its training data and can inherit biases from that data.

This project detects patterns in text and is not a medical or psychological diagnostic system.

## License

For academic/demo use. Add a license file appropriate to your own intended distribution.
