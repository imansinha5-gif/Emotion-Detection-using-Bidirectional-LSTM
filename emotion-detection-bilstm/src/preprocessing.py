"""Shared text cleaning: used by training AND the website so they always match."""
import re


def preprocess_text(text):
    text = str(text).lower()                          # 1. lowercase
    text = re.sub(r"http\S+|www\.\S+", " ", text)     # 2. remove URLs
    text = re.sub(r"@\w+", " ", text)                 # 3. drop @mentions
    text = re.sub(r"#(\w+)", r"\1", text)             # 4. keep hashtag word
    text = re.sub(r"[^a-z0-9\s'!?]", " ", text)       # 5. keep ! ? ' (emotion cues)
    return re.sub(r"\s+", " ", text).strip()          # 6. collapse whitespace
