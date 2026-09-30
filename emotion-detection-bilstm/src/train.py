"""Train the BiLSTM emotion classifier.
Run from project root:
    python -m src.train --data data/emotion_dataset.csv
"""
import os, json, argparse
import joblib, numpy as np, pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, callbacks
from src.preprocessing import preprocess_text

SEED, MAX_VOCAB, MAX_LEN = 42, 20000, 100
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(MAX_VOCAB, 128),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/emotion_dataset.csv")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()

    df = pd.read_csv(a.data)
    if not {"text", "emotion"}.issubset(df.columns):
        raise SystemExit(f"CSV must have columns 'text' and 'emotion'; found {list(df.columns)}")
    df = df.dropna(subset=["text", "emotion"]).copy()
    df["clean"] = df["text"].apply(preprocess_text)

    enc = LabelEncoder()
    y = enc.fit_transform(df["emotion"].astype(str))
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        df["clean"], y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )

    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tok.fit_on_texts(X_tr)

    def prep(s):
        return pad_sequences(
            tok.texts_to_sequences(s), maxlen=MAX_LEN, padding="post", truncating="post"
        )

    Xtr, Xva, Xte = prep(X_tr), prep(X_va), prep(X_te)
    model = build_model(len(enc.classes_))

    os.makedirs("models", exist_ok=True)
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint("models/emotion_bilstm.keras", monitor="val_loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        Xtr, y_tr, validation_data=(Xva, y_va),
        epochs=a.epochs, batch_size=a.batch, callbacks=cbs, verbose=1
    )

    loss, acc = model.evaluate(Xte, y_te, verbose=0)
    pred = model.predict(Xte, verbose=0).argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    report = classification_report(
        y_te, pred, target_names=enc.classes_, output_dict=True, zero_division=0
    )

    joblib.dump(tok, "models/tokenizer.pkl")
    joblib.dump(enc, "models/label_encoder.pkl")
    metrics = {
        "accuracy": round(float(acc) * 100, 2),
        "macro_f1": round(float(f1) * 100, 2),
        "dataset_size": int(len(df)),
        "num_classes": int(len(enc.classes_)),
        "classes": [str(x) for x in enc.classes_],
        "max_sequence_length": MAX_LEN,
        "vocab_size": min(MAX_VOCAB, len(tok.word_index) + 1),
    }
    json.dump(metrics, open("models/metrics.json", "w"), indent=2)
    json.dump(
        {"accuracy": history.history.get("accuracy", []),
         "val_accuracy": history.history.get("val_accuracy", []),
         "loss": history.history.get("loss", []),
         "val_loss": history.history.get("val_loss", [])},
        open("models/history.json", "w"), indent=2
    )
    json.dump(report, open("models/classification_report.json", "w"), indent=2)
    np.save("models/confusion_matrix.npy", confusion_matrix(y_te, pred))

    print(f"\nTest accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_te, pred, target_names=enc.classes_, zero_division=0))
    print("Saved model artifacts in models/")

if __name__ == "__main__":
    main()
