"""Fine-tune ModernBERT-base for the two heads.

Two independent fine-tunes (multi-label emotions, VAD regression) rather
than one multi-task model — simpler to debug; multi-task is left as an
optional ablation, not a v1 requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from lyric_emotion.config import Config
from lyric_emotion.data import GOEMOTIONS_LABELS

VAD_DIMS = ["valence", "arousal", "dominance"]


@dataclass
class TrainResult:
    head: str
    model_dir: Path
    metrics: dict[str, Any]


class _EncodedDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: torch.Tensor) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _tokenize(tokenizer: Any, texts: Any, max_length: int) -> dict[str, torch.Tensor]:
    return dict(
        tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    )


def _split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return df[df["split"] == split_name].reset_index(drop=True)


def _run_trainer(
    config: Config,
    model: Any,
    tokenizer: Any,
    train_ds: Dataset,
    val_ds: Dataset,
    model_dir: Path,
    epochs: int | None,
) -> Trainer:
    args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=epochs if epochs is not None else config.training.epochs,
        bf16=config.training.precision == "bf16" and torch.cuda.is_available(),
        logging_steps=50,
        seed=config.project.random_seed,
        save_total_limit=1,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return trainer


def _tune_thresholds(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[list[float], float, list[float]]:
    """Per-class threshold maximizing F1 on validation — 0.5 is not a safe
    default with GoEmotions' class imbalance (grief/relief have <200
    positives out of 43k train rows).
    """
    thresholds: list[float] = []
    per_class_f1: list[float] = []
    for i in range(probs.shape[1]):
        col_labels = labels[:, i]
        if col_labels.sum() == 0:
            thresholds.append(0.5)
            per_class_f1.append(0.0)
            continue
        precision, recall, candidate_thresholds = precision_recall_curve(col_labels, probs[:, i])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        has_candidates = len(candidate_thresholds) > 0
        best_idx = int(np.argmax(f1_scores[:-1])) if has_candidates else 0
        best_threshold = float(candidate_thresholds[best_idx]) if has_candidates else 0.5
        thresholds.append(best_threshold)
        per_class_f1.append(float(f1_scores[best_idx]) if has_candidates else 0.0)
    preds = (probs >= np.array(thresholds)).astype(int)
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return thresholds, macro_f1, per_class_f1


def train_emotions(
    config: Config, subsample: int | None = None, epochs: int | None = None
) -> TrainResult:
    set_seed(config.project.random_seed)
    df = pd.read_parquet(Path(config.paths.processed_dir) / "emotions_train.parquet")
    if subsample:
        df = df.groupby("split", group_keys=False).head(subsample)

    train_df, val_df = _split(df, "train"), _split(df, "validation")
    max_length = 256 if subsample else config.training.max_length

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    train_enc = _tokenize(tokenizer, train_df["text"], max_length)
    val_enc = _tokenize(tokenizer, val_df["text"], max_length)
    train_labels = torch.tensor(train_df[GOEMOTIONS_LABELS].to_numpy(), dtype=torch.float)
    val_labels = torch.tensor(val_df[GOEMOTIONS_LABELS].to_numpy(), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.training.base_model,
        num_labels=len(GOEMOTIONS_LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(GOEMOTIONS_LABELS)),
        label2id={label: i for i, label in enumerate(GOEMOTIONS_LABELS)},
    )

    model_dir = Path("models") / "emotions"
    trainer = _run_trainer(
        config,
        model,
        tokenizer,
        _EncodedDataset(train_enc, train_labels),
        _EncodedDataset(val_enc, val_labels),
        model_dir,
        epochs,
    )

    val_logits = trainer.predict(_EncodedDataset(val_enc, val_labels)).predictions
    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
    thresholds, macro_f1, per_class_f1 = _tune_thresholds(val_probs, val_labels.numpy())

    metrics = {
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(GOEMOTIONS_LABELS, per_class_f1, strict=True)),
        "thresholds": dict(zip(GOEMOTIONS_LABELS, thresholds, strict=True)),
        "n_train": len(train_df),
        "n_val": len(val_df),
    }
    return TrainResult("emotions", model_dir, metrics)


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Concordance correlation coefficient — stricter than Pearson r since
    it also penalizes location/scale shift, standard for VAD regression.
    """
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    return float(2 * covariance / (var_x + var_y + (mean_x - mean_y) ** 2 + 1e-9))


def train_vad(
    config: Config, subsample: int | None = None, epochs: int | None = None
) -> TrainResult:
    set_seed(config.project.random_seed)
    df = pd.read_parquet(Path(config.paths.processed_dir) / "vad_train.parquet")
    if subsample:
        df = df.groupby("split", group_keys=False).head(subsample)

    train_df, val_df = _split(df, "train"), _split(df, "dev")
    max_length = 256 if subsample else config.training.max_length

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    train_enc = _tokenize(tokenizer, train_df["text"], max_length)
    val_enc = _tokenize(tokenizer, val_df["text"], max_length)
    train_labels = torch.tensor(train_df[VAD_DIMS].to_numpy(), dtype=torch.float)
    val_labels = torch.tensor(val_df[VAD_DIMS].to_numpy(), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.training.base_model,
        num_labels=len(VAD_DIMS),
        problem_type="regression",
        id2label=dict(enumerate(VAD_DIMS)),
        label2id={dim: i for i, dim in enumerate(VAD_DIMS)},
    )

    model_dir = Path("models") / "vad"
    trainer = _run_trainer(
        config,
        model,
        tokenizer,
        _EncodedDataset(train_enc, train_labels),
        _EncodedDataset(val_enc, val_labels),
        model_dir,
        epochs,
    )

    val_preds: np.ndarray = np.asarray(
        trainer.predict(_EncodedDataset(val_enc, val_labels)).predictions
    )
    val_labels_np = val_labels.numpy()

    per_dim_ccc = {
        dim: _ccc(val_preds[:, i], val_labels_np[:, i]) for i, dim in enumerate(VAD_DIMS)
    }
    per_dim_pearson = {
        dim: float(np.corrcoef(val_preds[:, i], val_labels_np[:, i])[0, 1])
        for i, dim in enumerate(VAD_DIMS)
    }

    metrics = {
        "ccc": per_dim_ccc,
        "pearson_r": per_dim_pearson,
        "n_train": len(train_df),
        "n_val": len(val_df),
    }
    return TrainResult("vad", model_dir, metrics)
