"""Baseline comparisons required by the Phase 2 acceptance criteria:
`cirimus/modernbert-base-go-emotions` for the emotions head, and the
NRC-VAD lexicon for the VAD head. Both run on CPU — one-off eval over a
validation split, not worth competing with the training run for GPU memory.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lyric_emotion.config import Config
from lyric_emotion.data import GOEMOTIONS_LABELS
from lyric_emotion.model import VAD_DIMS, _ccc, _tune_thresholds

CIRIMUS_MODEL = "cirimus/modernbert-base-go-emotions"

# Non-commercial research use only (NRC terms of use) — never committed to
# the repo; downloaded on demand into the gitignored cache dir.
NRC_VAD_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
NRC_VAD_MEMBER = "NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"
_WORD_RE = re.compile(r"[a-zA-Z']+")


def evaluate_emotions_baseline(config: Config) -> dict:
    """Score cirimus/modernbert-base-go-emotions on our GoEmotions
    validation split the same way we score our own model, for a fair
    macro-F1 comparison.
    """
    df = pd.read_parquet(Path(config.paths.processed_dir) / "emotions_train.parquet")
    val_df = df[df["split"] == "validation"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(CIRIMUS_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CIRIMUS_MODEL)
    model.eval()

    label_order = [model.config.id2label[i] for i in range(model.config.num_labels)]
    unmapped = [label for label in label_order if label not in GOEMOTIONS_LABELS]

    probs = np.zeros((len(val_df), len(GOEMOTIONS_LABELS)), dtype=np.float32)
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(val_df), batch_size):
            batch = val_df["text"].iloc[start : start + batch_size].tolist()
            enc = tokenizer(
                batch, truncation=True, padding=True, max_length=512, return_tensors="pt"
            )
            batch_probs = torch.sigmoid(model(**enc).logits).numpy()
            for i, label in enumerate(label_order):
                if label in GOEMOTIONS_LABELS:
                    col = GOEMOTIONS_LABELS.index(label)
                    probs[start : start + len(batch), col] = batch_probs[:, i]

    labels = val_df[GOEMOTIONS_LABELS].to_numpy()
    thresholds, macro_f1, per_class_f1 = _tune_thresholds(probs, labels)
    return {
        "model": CIRIMUS_MODEL,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(GOEMOTIONS_LABELS, per_class_f1, strict=True)),
        "n_val": len(val_df),
        "unmapped_labels": unmapped,
    }


def _download_nrc_vad_lexicon(cache_dir: Path) -> Path:
    dest = cache_dir / "nrc_vad_unigrams.tsv"
    if dest.exists():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(NRC_VAD_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        dest.write_bytes(zf.read(NRC_VAD_MEMBER))
    return dest


def _load_nrc_vad_lexicon(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t").set_index("term")


def _lexicon_vad_scores(texts: pd.Series, lexicon: pd.DataFrame) -> np.ndarray:
    """Mean VAD of the lexicon words found in each text; 0 (neutral) for
    texts with no matching words.
    """
    values = lexicon[VAD_DIMS].to_numpy()
    word_to_row = {word: i for i, word in enumerate(lexicon.index)}
    scores = np.zeros((len(texts), len(VAD_DIMS)), dtype=np.float32)
    for i, text in enumerate(texts):
        rows = [word_to_row[w] for w in _WORD_RE.findall(text.lower()) if w in word_to_row]
        if rows:
            scores[i] = values[rows].mean(axis=0)
    return scores


def evaluate_vad_baseline(config: Config) -> dict:
    """NRC-VAD lexicon baseline: average word-level VAD scores per text,
    scored against the EmoBank dev split with the same CCC/Pearson metrics
    as our fine-tuned model.
    """
    df = pd.read_parquet(Path(config.paths.processed_dir) / "vad_train.parquet")
    val_df = df[df["split"] == "dev"].reset_index(drop=True)

    lexicon_path = _download_nrc_vad_lexicon(Path(config.paths.cache_dir))
    lexicon = _load_nrc_vad_lexicon(lexicon_path)

    preds = _lexicon_vad_scores(val_df["text"], lexicon)
    labels = val_df[VAD_DIMS].to_numpy()

    per_dim_ccc = {dim: _ccc(preds[:, i], labels[:, i]) for i, dim in enumerate(VAD_DIMS)}
    per_dim_pearson = {
        dim: float(np.corrcoef(preds[:, i], labels[:, i])[0, 1]) for i, dim in enumerate(VAD_DIMS)
    }
    return {
        "model": "NRC-VAD lexicon (mean word score)",
        "ccc": per_dim_ccc,
        "pearson_r": per_dim_pearson,
        "n_val": len(val_df),
    }


def write_baseline_report(
    our_emotions: dict, baseline_emotions: dict, our_vad: dict, baseline_vad: dict, docs_dir: Path
) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline comparison\n",
        "## Emotions head (macro-F1, tuned thresholds, GoEmotions validation split)\n",
        "| model | macro F1 | n |",
        "|---|---|---|",
        f"| lyric-emotion (ours) | {our_emotions['macro_f1']:.4f} | {our_emotions['n_val']} |",
        f"| {baseline_emotions['model']} | {baseline_emotions['macro_f1']:.4f} "
        f"| {baseline_emotions['n_val']} |",
        "",
        "## VAD head (CCC / Pearson r, EmoBank dev split)\n",
        "| model | dim | CCC | Pearson r |",
        "|---|---|---|---|",
    ]
    for dim in VAD_DIMS:
        lines.append(
            f"| lyric-emotion (ours) | {dim} | {our_vad['ccc'][dim]:.4f} "
            f"| {our_vad['pearson_r'][dim]:.4f} |"
        )
    for dim in VAD_DIMS:
        lines.append(
            f"| {baseline_vad['model']} | {dim} | {baseline_vad['ccc'][dim]:.4f} "
            f"| {baseline_vad['pearson_r'][dim]:.4f} |"
        )
    lines.append(f"\nn = {baseline_vad['n_val']}")
    path = docs_dir / "baseline_comparison.md"
    path.write_text("\n".join(lines))
    return path
