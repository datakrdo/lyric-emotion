"""Baseline comparisons required by the Phase 2 acceptance criteria:
`cirimus/modernbert-base-go-emotions` for the emotions head, and the
NRC-VAD lexicon for the VAD head. The NRC lexicon baseline is pure numpy
(CPU); cirimus uses the GPU if available, same as our own model.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lyric_emotion.config import Config
from lyric_emotion.data import GOEMOTIONS_LABELS, drop_split_leaks
from lyric_emotion.model import (
    VAD_DIMS,
    _ccc,
    _tune_thresholds,
    score_multilabel,
    score_vad,
)

CIRIMUS_MODEL = "cirimus/modernbert-base-go-emotions"

# Non-commercial research use only (NRC terms of use) — never committed to
# the repo; downloaded on demand into the gitignored cache dir.
NRC_VAD_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
NRC_VAD_MEMBER = "NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _score_cirimus(tokenizer: Any, model: Any, texts: pd.Series) -> tuple[np.ndarray, list[str]]:
    label_order = [model.config.id2label[i] for i in range(model.config.num_labels)]
    unmapped = [label for label in label_order if label not in GOEMOTIONS_LABELS]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    probs = np.zeros((len(texts), len(GOEMOTIONS_LABELS)), dtype=np.float32)
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts.iloc[start : start + batch_size].tolist()
            enc = tokenizer(
                batch, truncation=True, padding=True, max_length=512, return_tensors="pt"
            ).to(device)
            batch_probs = torch.sigmoid(model(**enc).logits).cpu().numpy()
            for i, label in enumerate(label_order):
                if label in GOEMOTIONS_LABELS:
                    col = GOEMOTIONS_LABELS.index(label)
                    probs[start : start + len(batch), col] = batch_probs[:, i]
    return probs, unmapped


def evaluate_emotions_baseline(config: Config) -> dict:
    """Score cirimus/modernbert-base-go-emotions on our GoEmotions splits the
    same way we score our own model: thresholds tuned on validation,
    macro/micro-F1 and mAP reported on the held-out test split.
    """
    df = pd.read_parquet(Path(config.paths.processed_dir) / "emotions_train.parquet")
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(CIRIMUS_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CIRIMUS_MODEL)
    model.eval()

    val_probs, unmapped = _score_cirimus(tokenizer, model, val_df["text"])
    thresholds, _, _ = _tune_thresholds(val_probs, val_df[GOEMOTIONS_LABELS].to_numpy())

    test_probs, _ = _score_cirimus(tokenizer, model, test_df["text"])
    test_metrics = score_multilabel(test_probs, test_df[GOEMOTIONS_LABELS].to_numpy(), thresholds)

    return {
        "model": CIRIMUS_MODEL,
        **test_metrics,
        "n_val": len(val_df),
        "n_test": len(test_df),
        "unmapped_labels": unmapped,
    }


def _download_nrc_vad_lexicon(cache_dir: Path) -> Path:
    dest = cache_dir / "nrc_vad_unigrams.tsv"
    if dest.exists():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (research download; lyric-emotion)"}
    response = requests.get(NRC_VAD_URL, timeout=60, headers=headers)
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
    scored against the EmoBank test split with the same CCC/Pearson metrics
    as our fine-tuned model (which is trained on train, early-stopped on dev).
    """
    df = pd.read_parquet(Path(config.paths.processed_dir) / "vad_train.parquet")
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    lexicon_path = _download_nrc_vad_lexicon(Path(config.paths.cache_dir))
    lexicon = _load_nrc_vad_lexicon(lexicon_path)

    preds = _lexicon_vad_scores(test_df["text"], lexicon)
    labels = test_df[VAD_DIMS].to_numpy()

    per_dim_ccc = {dim: _ccc(preds[:, i], labels[:, i]) for i, dim in enumerate(VAD_DIMS)}
    per_dim_pearson = {
        dim: float(np.corrcoef(preds[:, i], labels[:, i])[0, 1]) for i, dim in enumerate(VAD_DIMS)
    }
    return {
        "model": "NRC-VAD lexicon (mean word score)",
        "ccc": per_dim_ccc,
        "pearson_r": per_dim_pearson,
        "n_test": len(test_df),
    }


def _score_vad_checkpoint(
    config: Config, model_dir: str, texts: pd.Series, unit: str
) -> np.ndarray:
    """Score `texts` with a VAD checkpoint, whole-lyric or by sections."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return score_vad(model, tokenizer, list(texts), device, config.training.max_length, 32, unit)


def evaluate_vad_checkpoints(config: Config) -> dict:
    """Compare the stage-1 (`vad`) and stage-2 domain-adapted (`vad_domain`)
    checkpoints, each scoring whole lyrics and by sections, on the DMDD test
    split — real song lyrics with valence/arousal labels. Test texts that
    also appear in train/validation are dropped first. The winner sets
    `config.models.case_study_vad` / `case_study_vad_unit`. Dominance has no
    DMDD label, so only valence/arousal are compared.
    """
    df = drop_split_leaks(
        pd.read_parquet(Path(config.paths.processed_dir) / "domain_train.parquet")
    )
    test_df = df[(df["source"] == "dmdd") & (df["split"] == "test")].reset_index(drop=True)
    labels = test_df[["valence", "arousal"]].to_numpy()

    def metrics(preds: np.ndarray) -> dict:
        dims = ["valence", "arousal"]
        return {
            "ccc": {d: _ccc(preds[:, i], labels[:, i]) for i, d in enumerate(dims)},
            "pearson_r": {
                d: float(np.corrcoef(preds[:, i], labels[:, i])[0, 1]) for i, d in enumerate(dims)
            },
        }

    results: dict[str, Any] = {}
    checkpoints = {"vad": config.models.vad_dir, "vad_domain": config.models.vad_domain_dir}
    for name, model_dir in checkpoints.items():
        for unit in ("full", "sections"):
            preds = _score_vad_checkpoint(config, model_dir, test_df["text"], unit)[:, :2]
            results[f"{name}/{unit}"] = metrics(preds)
    # What `analyze` actually scores with: the mean of the per-seed checkpoints.
    members = config.models.case_study_vad_ensemble
    if members:
        unit = config.models.case_study_vad_unit
        runs = [_score_vad_checkpoint(config, d, test_df["text"], unit)[:, :2] for d in members]
        results[f"vad_domain_mean_of_{len(members)}_seeds/{unit}"] = metrics(np.mean(runs, axis=0))
    results["n_test"] = len(test_df)
    return results


def write_vad_checkpoint_report(result: dict, docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VAD checkpoint comparison (stage 1 vs stage 2, DMDD test split)\n",
        "Decides which checkpoint and scoring unit `config.models.case_study_vad` /",
        "`case_study_vad_unit` should use, by scoring both checkpoints on real song",
        "lyrics with valence/arousal labels, whole-lyric (`full`) and by sections.",
        "Test texts that also appear in train/validation are excluded.\n",
        "| checkpoint | unit | dim | CCC | Pearson r |",
        "|---|---|---|---|---|",
    ]
    for key in (k for k in result if "/" in k):
        name, unit = key.split("/")
        for dim in ("valence", "arousal"):
            lines.append(
                f"| {name} | {unit} | {dim} | {result[key]['ccc'][dim]:.4f} "
                f"| {result[key]['pearson_r'][dim]:.4f} |"
            )
    lines.append(f"\nn = {result['n_test']}")
    path = docs_dir / "vad_checkpoint_comparison.md"
    path.write_text("\n".join(lines))
    return path


def emotion_vad_consistency(config: Config, song_scores: pd.DataFrame) -> dict:
    """Fase 4 label-free consistency check: project each song's 28 emotion
    probabilities to VAD via NRC-VAD-lexicon centroids (the lexicon score of
    the emotion word itself) and correlate against the regression head's own
    VAD predictions on the same songs. Low correlation flags a head that
    disagrees with itself — a problem the (label-free) gold set can't yet
    catch since it isn't annotated.
    """
    lexicon_path = _download_nrc_vad_lexicon(Path(config.paths.cache_dir))
    lexicon = _load_nrc_vad_lexicon(lexicon_path)

    centroid_labels = [label for label in GOEMOTIONS_LABELS if label in lexicon.index]
    missing_labels = [label for label in GOEMOTIONS_LABELS if label not in lexicon.index]
    centroids = lexicon.loc[centroid_labels, VAD_DIMS].to_numpy()  # (n_labels, 3)

    probs = song_scores[[f"prob_{label}" for label in centroid_labels]].to_numpy()
    weights_sum = probs.sum(axis=1, keepdims=True)
    projected = np.divide(
        probs @ centroids, weights_sum, out=np.zeros((len(probs), 3)), where=weights_sum > 0
    )

    pearson_r = {
        dim: float(np.corrcoef(projected[:, i], song_scores[dim].to_numpy())[0, 1])
        for i, dim in enumerate(VAD_DIMS)
    }
    return {
        "pearson_r": pearson_r,
        "n_songs": len(song_scores),
        "missing_labels": missing_labels,
    }


def write_consistency_report(result: dict, docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Emotion -> VAD consistency check (label-free, Phase 4)\n",
        "Projects each song's 28 emotion probabilities to VAD using NRC-VAD-lexicon",
        "centroids, then correlates against the regression head's own VAD output",
        "on the same songs. No gold-set labels involved.\n",
        "| dim | Pearson r (projected vs regression head) |",
        "|---|---|",
    ]
    for dim in VAD_DIMS:
        lines.append(f"| {dim} | {result['pearson_r'][dim]:.4f} |")
    lines.append(f"\nn songs = {result['n_songs']}")
    if result["missing_labels"]:
        missing = result["missing_labels"]
        lines.append(f"\nEmotion labels not found in the NRC-VAD lexicon: {missing}")
    path = docs_dir / "consistency_check.md"
    path.write_text("\n".join(lines))
    return path


def write_baseline_report(
    our_emotions: dict, baseline_emotions: dict, our_vad: dict, baseline_vad: dict, docs_dir: Path
) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline comparison\n",
        "## Emotions head (thresholds tuned on validation, scored on the GoEmotions test split)\n",
        "| model | macro F1 | micro F1 | mAP | n test |",
        "|---|---|---|---|---|",
        f"| lyric-emotion (ours) | {our_emotions['macro_f1']:.4f} "
        f"| {our_emotions['micro_f1']:.4f} | {our_emotions['mean_average_precision']:.4f} "
        f"| {our_emotions['n_test']} |",
        f"| {baseline_emotions['model']} | {baseline_emotions['macro_f1']:.4f} "
        f"| {baseline_emotions['micro_f1']:.4f} "
        f"| {baseline_emotions['mean_average_precision']:.4f} "
        f"| {baseline_emotions['n_test']} |",
        "",
        "## VAD head (CCC / Pearson r, EmoBank test split)\n",
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
    lines.append(f"\nn = {baseline_vad['n_test']}")
    lines.append(
        "\nBoth models trained on the same train split and validated on the "
        "same held-out split their own thresholds/checkpoints were tuned "
        "against — not the same split used to pick the threshold or the "
        "early-stopping checkpoint."
    )
    path = docs_dir / "baseline_comparison.md"
    path.write_text("\n".join(lines))
    return path


def _icc_2_1(ratings_a: np.ndarray, ratings_b: np.ndarray) -> float:
    """Two-way random, single-rater, absolute-agreement ICC(2,1) between two
    sets of ratings on the same items (one annotator, two sessions here).
    """
    data = np.stack([ratings_a, ratings_b], axis=1)  # (n_items, 2 raters)
    n, k = data.shape
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ms_rows = k * ((row_means - grand_mean) ** 2).sum() / (n - 1)
    ms_cols = n * ((col_means - grand_mean) ** 2).sum() / (k - 1)
    ms_error = ((data - row_means[:, None] - col_means[None, :] + grand_mean) ** 2).sum() / (
        (n - 1) * (k - 1)
    )
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom == 0:
        return 0.0
    return float((ms_rows - ms_error) / denom)
