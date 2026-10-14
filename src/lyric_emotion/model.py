"""Fine-tune ModernBERT-base for the two heads.

Two independent fine-tunes (multi-label emotions, VAD regression) rather
than one multi-task model — simpler to debug; multi-task is left as an
optional ablation, not a v1 requirement.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from torch.utils.data import Dataset, SequentialSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from lyric_emotion.config import Config
from lyric_emotion.data import GOEMOTIONS_LABELS, split_sections

VAD_DIMS = ["valence", "arousal", "dominance"]


@dataclass
class TrainResult:
    head: str
    model_dir: Path
    metrics: dict[str, Any]


class _EncodedDataset(Dataset):
    """Holds un-padded token id lists; padding happens per-batch via
    `DataCollatorWithPadding` instead of up front, so a short GoEmotions
    comment (~20 tokens) doesn't pay the cost of a 1024-token sequence.
    """

    def __init__(self, encodings: dict[str, Any], labels: torch.Tensor) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class _PaddingCollator:
    """`DataCollatorWithPadding` pads every feature it's given, including
    `labels` — fine for single-class labels, but ours are fixed-width
    vectors (28 emotions, or valence/arousal/dominance) that just need
    stacking, not padding. Pop them out, pad the rest, stack back in.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = torch.stack([f.pop("labels") for f in features])
        batch = self._pad(features)
        batch["labels"] = labels
        return batch


def _tokenize(tokenizer: Any, texts: Any, max_length: int) -> dict[str, Any]:
    return dict(tokenizer(list(texts), truncation=True, max_length=max_length))


def _split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return df[df["split"] == split_name].reset_index(drop=True)


class _SequentialEvalTrainer(Trainer):
    """`train_sampling_strategy="group_by_length"` also reorders eval/predict
    batches by default (`Trainer._get_eval_sampler` reuses the same length
    grouping) — fine for training, but it breaks our val/test code, which
    zips `trainer.predict(...).predictions` back up against label arrays
    built in the original dataframe order. Pin eval back to dataset order.
    """

    def _get_eval_sampler(self, eval_dataset: Any) -> torch.utils.data.Sampler:
        return SequentialSampler(eval_dataset)


class _WeightedBCETrainer(_SequentialEvalTrainer):
    """Trainer that replaces the model's default (unweighted) BCE loss with
    one using per-class pos_weight, so rare labels (e.g. GoEmotions' grief/
    relief, <200 positives out of 43k rows) aren't cheaply ignored just
    because predicting all-negative already yields low average loss.
    """

    def __init__(self, *args: Any, pos_weight: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def _training_args(
    config: Config, model_dir: Path, epochs: int | None, gradient_checkpointing: bool | None = None
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=epochs if epochs is not None else config.training.epochs,
        bf16=config.training.precision == "bf16" and torch.cuda.is_available(),
        gradient_checkpointing=(
            gradient_checkpointing
            if gradient_checkpointing is not None
            else config.training.gradient_checkpointing
        ),
        # Batches group same-length sequences so padding tracks the batch's
        # own max length, not the global max_length cap.
        train_sampling_strategy="group_by_length",
        logging_steps=50,
        seed=config.project.random_seed,
        save_total_limit=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )


def _run_trainer(
    config: Config,
    model: Any,
    tokenizer: Any,
    train_ds: Dataset,
    val_ds: Dataset,
    model_dir: Path,
    epochs: int | None,
    pos_weight: torch.Tensor | None = None,
    resume: bool = False,
    trainer_cls: type[Trainer] = _SequentialEvalTrainer,
    trainer_kwargs: dict[str, Any] | None = None,
    gradient_checkpointing: bool | None = None,
) -> Trainer:
    args = _training_args(config, model_dir, epochs, gradient_checkpointing)
    kwargs: dict[str, Any] = dict(trainer_kwargs or {})
    if pos_weight is not None:
        trainer_cls = _WeightedBCETrainer
        kwargs["pos_weight"] = pos_weight
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=_PaddingCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        **kwargs,
    )
    trainer.train(resume_from_checkpoint=resume)
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


def score_multilabel(
    probs: np.ndarray, labels: np.ndarray, thresholds: list[float]
) -> dict[str, Any]:
    """Macro/micro F1 (at fixed, already-tuned thresholds) plus per-class
    average precision (mAP) — the plan's Phase 2 metric set for the
    emotions head, evaluated on whatever split is passed in.
    """
    preds = (probs >= np.array(thresholds)).astype(int)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_ap = [
        float(average_precision_score(labels[:, i], probs[:, i])) if labels[:, i].sum() else 0.0
        for i in range(labels.shape[1])
    ]
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "mean_average_precision": float(np.mean(per_class_ap)),
        "per_class_f1": dict(zip(GOEMOTIONS_LABELS, per_class_f1.tolist(), strict=True)),
        "per_class_average_precision": dict(zip(GOEMOTIONS_LABELS, per_class_ap, strict=True)),
    }


def train_emotions(
    config: Config, subsample: int | None = None, epochs: int | None = None, resume: bool = False
) -> TrainResult:
    set_seed(config.project.random_seed)
    df = pd.read_parquet(Path(config.paths.processed_dir) / "emotions_train.parquet")
    if subsample:
        df = df.groupby("split", group_keys=False).head(subsample)

    train_df, val_df, test_df = _split(df, "train"), _split(df, "validation"), _split(df, "test")
    max_length = config.training.short_text_max_length

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    train_enc = _tokenize(tokenizer, train_df["text"], max_length)
    val_enc = _tokenize(tokenizer, val_df["text"], max_length)
    test_enc = _tokenize(tokenizer, test_df["text"], max_length)
    train_labels = torch.tensor(train_df[GOEMOTIONS_LABELS].to_numpy(), dtype=torch.float)
    val_labels = torch.tensor(val_df[GOEMOTIONS_LABELS].to_numpy(), dtype=torch.float)
    test_labels = torch.tensor(test_df[GOEMOTIONS_LABELS].to_numpy(), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.training.base_model,
        num_labels=len(GOEMOTIONS_LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(GOEMOTIONS_LABELS)),
        label2id={label: i for i, label in enumerate(GOEMOTIONS_LABELS)},
    )

    # Inverse-frequency pos_weight for BCE, capped at 20x so the rarest
    # labels (grief/relief, <200 positives) don't destabilize training with
    # an extreme multiplier.
    pos_counts = train_labels.sum(dim=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).clamp(max=20.0)

    model_dir = Path(config.models.emotions_dir)
    trainer = _run_trainer(
        config,
        model,
        tokenizer,
        _EncodedDataset(train_enc, train_labels),
        _EncodedDataset(val_enc, val_labels),
        model_dir,
        epochs,
        pos_weight=pos_weight,
        resume=resume,
    )

    # Thresholds are tuned on validation only; every reported metric is
    # scored on the held-out test split so it isn't inflated by fitting
    # thresholds and reporting on the same rows.
    val_logits = trainer.predict(_EncodedDataset(val_enc, val_labels)).predictions
    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
    thresholds, _, _ = _tune_thresholds(val_probs, val_labels.numpy())

    test_logits = trainer.predict(_EncodedDataset(test_enc, test_labels)).predictions
    test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()
    test_metrics = score_multilabel(test_probs, test_labels.numpy(), thresholds)

    metrics = {
        "thresholds": dict(zip(GOEMOTIONS_LABELS, thresholds, strict=True)),
        **test_metrics,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    return TrainResult("emotions", model_dir, metrics)


class _MaskedVADTrainer(_SequentialEvalTrainer):
    """MSE only on valence/arousal — domain weak labels (MoodyLyrics/DMDD)
    carry no dominance, so that column keeps its stage-1 weights untouched.
    """

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.mse_loss(outputs.logits[:, :2], labels[:, :2])
        return (loss, outputs) if return_outputs else loss


def train_vad_domain(
    config: Config,
    subsample: int | None = None,
    epochs: int | None = None,
    resume: bool = False,
    sources: tuple[str, ...] | None = None,
    seed: int | None = None,
    model_dir: Path | None = None,
) -> TrainResult:
    """Phase 4 stage-2: continue the VAD head from its stage-1 checkpoint on
    weak-labeled lyrics (MoodyLyrics + DMDD sample), valence/arousal only.
    `sources`, `seed` and `model_dir` exist for the ablation runs.
    """
    if seed is not None:
        config = config.model_copy(
            update={"project": config.project.model_copy(update={"random_seed": seed})}
        )
    set_seed(config.project.random_seed)
    df = pd.read_parquet(Path(config.paths.processed_dir) / "domain_train.parquet")
    if sources:
        df = df[df["source"].isin(sources)]
    if subsample:
        df = df.groupby("split", group_keys=False).head(subsample)

    # MoodyLyrics rows are all tagged "train" (no official split); DMDD
    # contributes the "validation"/"test" rows used for early stopping and
    # the final held-out report respectively.
    train_df, val_df, test_df = _split(df, "train"), _split(df, "validation"), _split(df, "test")
    max_length = config.training.max_length

    base_dir = config.models.vad_dir
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    train_enc = _tokenize(tokenizer, train_df["text"], max_length)
    val_enc = _tokenize(tokenizer, val_df["text"], max_length)
    test_enc = _tokenize(tokenizer, test_df["text"], max_length)

    def _va_labels(split_df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(
            split_df[["valence", "arousal"]].assign(dominance=0.0).to_numpy(), dtype=torch.float
        )

    train_labels, val_labels, test_labels = (
        _va_labels(train_df),
        _va_labels(val_df),
        _va_labels(test_df),
    )

    model = AutoModelForSequenceClassification.from_pretrained(base_dir)

    model_dir = model_dir or Path(config.models.vad_domain_dir)
    trainer = _run_trainer(
        config,
        model,
        tokenizer,
        _EncodedDataset(train_enc, train_labels),
        _EncodedDataset(val_enc, val_labels),
        model_dir,
        epochs,
        resume=resume,
        trainer_cls=_MaskedVADTrainer,
        # Whole lyric sheets, not short GoEmotions/EmoBank text — worth the
        # recompute-vs-memory trade gradient checkpointing makes here, even
        # though it's off by default for the other two (short-text) heads.
        gradient_checkpointing=True,
    )

    test_preds: np.ndarray = np.asarray(
        trainer.predict(_EncodedDataset(test_enc, test_labels)).predictions
    )
    test_labels_np = test_labels.numpy()

    per_dim_ccc = {
        dim: _ccc(test_preds[:, i], test_labels_np[:, i])
        for i, dim in enumerate(["valence", "arousal"])
    }
    per_dim_pearson = {
        dim: float(np.corrcoef(test_preds[:, i], test_labels_np[:, i])[0, 1])
        for i, dim in enumerate(["valence", "arousal"])
    }

    metrics = {
        "ccc": per_dim_ccc,
        "pearson_r": per_dim_pearson,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "note": "dominance not adapted — weak labels carry no dominance signal",
    }
    return TrainResult("vad_domain", model_dir, metrics)


def read_train_metrics(report_path: Path) -> dict[str, Any]:
    """Parse the JSON block out of a docs/training_<head>.md report."""
    import json

    block = report_path.read_text().split("```json\n")[1].split("\n```")[0]
    return json.loads(block)  # type: ignore[no-any-return]


def _score_batches(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: str,
    max_length: int,
    batch_size: int,
    apply_sigmoid: bool,
) -> np.ndarray:
    """Run `model` over `texts` in batches, sorted by token length so each
    batch pads to its own (short) max rather than the global cap — the
    inference-time equivalent of `group_by_length` training.
    """
    encoded = [tokenizer(t, truncation=True, max_length=max_length) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(encoded[i]["input_ids"]))
    use_bf16 = device == "cuda"

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    with torch.no_grad(), autocast_ctx:
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = tokenizer.pad([encoded[i] for i in idx], return_tensors="pt").to(device)
            logits = model(**batch).logits
            if apply_sigmoid:
                logits = torch.sigmoid(logits)
            out[idx] = logits.float().cpu().numpy()
    return out


def _mean_by_song(section_scores: np.ndarray, song_index: np.ndarray, n_songs: int) -> np.ndarray:
    """Average per-section scores back up to one row per song."""
    out = np.zeros((n_songs, section_scores.shape[1]), dtype=np.float32)
    counts = np.bincount(song_index, minlength=n_songs)
    np.add.at(out, song_index, section_scores)
    return out / counts[:, None]


def score_vad(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: str,
    max_length: int,
    batch_size: int,
    unit: str,
) -> np.ndarray:
    """VAD for each text, scoring the whole lyric ("full") or its sections
    averaged back to one row per song ("sections")."""
    if unit == "full":
        return _score_batches(model, tokenizer, list(texts), device, max_length, batch_size, False)
    sections_per_text = [split_sections(t) for t in texts]
    flat = [s for sections in sections_per_text for s in sections]
    index = np.repeat(np.arange(len(texts)), [len(s) for s in sections_per_text])
    scores = _score_batches(model, tokenizer, flat, device, max_length, batch_size, False)
    return _mean_by_song(scores, index, len(texts))


def predict_texts(config: Config, texts: list[str], docs_dir: Path = Path("docs")) -> pd.DataFrame:
    """Score lyrics text with both trained heads: per-emotion probability +
    threshold-based label, and continuous valence/arousal/dominance.

    Each song is split into its labeled sections (`split_sections`, which
    also collapses a repeated chorus to one occurrence) and the emotions head scores
    section-by-section, averaged back up to one row per song; the VAD head scores
    whole lyrics or sections per `models.case_study_vad_unit` — closer
    to the ~20-100 token spans both heads were trained on than a full lyric
    sheet fed in as a single sequence.
    """
    thresholds = read_train_metrics(docs_dir / "training_emotions.md")["thresholds"]
    threshold_vec = np.array([thresholds[label] for label in GOEMOTIONS_LABELS])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emotions_dir = config.models.emotions_dir
    vad_checkpoints = {"vad": config.models.vad_dir, "vad_domain": config.models.vad_domain_dir}
    vad_dir = vad_checkpoints[config.models.case_study_vad]

    emotions_tokenizer = AutoTokenizer.from_pretrained(emotions_dir)
    emotions_model = AutoModelForSequenceClassification.from_pretrained(emotions_dir).to(device)
    emotions_model.eval()

    max_length = config.training.max_length
    batch_size = 32

    sections_per_song = [split_sections(t) for t in texts]
    flat_texts = [s for sections in sections_per_song for s in sections]
    song_index = np.repeat(np.arange(len(texts)), [len(s) for s in sections_per_song])

    section_emotion_probs = _score_batches(
        emotions_model, emotions_tokenizer, flat_texts, device, max_length, batch_size, True
    )

    emotion_probs = _mean_by_song(section_emotion_probs, song_index, len(texts))
    vad_dirs = config.models.case_study_vad_ensemble or [vad_dir]
    vad_runs = []
    for member_dir in vad_dirs:
        vad_tokenizer = AutoTokenizer.from_pretrained(member_dir)
        vad_model = AutoModelForSequenceClassification.from_pretrained(member_dir).to(device)
        vad_model.eval()
        vad_runs.append(
            score_vad(
                vad_model,
                vad_tokenizer,
                texts,
                device,
                max_length,
                batch_size,
                config.models.case_study_vad_unit,
            )
        )
    vad_preds = np.mean(vad_runs, axis=0)

    result = pd.DataFrame({"text": texts})
    for i, label in enumerate(GOEMOTIONS_LABELS):
        result[f"prob_{label}"] = emotion_probs[:, i]
        result[label] = (emotion_probs[:, i] >= threshold_vec[i]).astype(int)
    for i, dim in enumerate(VAD_DIMS):
        result[dim] = vad_preds[:, i]
    return result


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Concordance correlation coefficient — stricter than Pearson r since
    it also penalizes location/scale shift, standard for VAD regression.
    """
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    return float(2 * covariance / (var_x + var_y + (mean_x - mean_y) ** 2 + 1e-9))


def train_vad(
    config: Config, subsample: int | None = None, epochs: int | None = None, resume: bool = False
) -> TrainResult:
    set_seed(config.project.random_seed)
    df = pd.read_parquet(Path(config.paths.processed_dir) / "vad_train.parquet")
    if subsample:
        df = df.groupby("split", group_keys=False).head(subsample)

    # "dev" trains/early-stops the model; "test" is held out for the final
    # reported numbers so they aren't measured on the same rows used to pick
    # the best checkpoint.
    train_df, val_df, test_df = _split(df, "train"), _split(df, "dev"), _split(df, "test")
    max_length = config.training.short_text_max_length

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    train_enc = _tokenize(tokenizer, train_df["text"], max_length)
    val_enc = _tokenize(tokenizer, val_df["text"], max_length)
    test_enc = _tokenize(tokenizer, test_df["text"], max_length)
    train_labels = torch.tensor(train_df[VAD_DIMS].to_numpy(), dtype=torch.float)
    val_labels = torch.tensor(val_df[VAD_DIMS].to_numpy(), dtype=torch.float)
    test_labels = torch.tensor(test_df[VAD_DIMS].to_numpy(), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.training.base_model,
        num_labels=len(VAD_DIMS),
        problem_type="regression",
        id2label=dict(enumerate(VAD_DIMS)),
        label2id={dim: i for i, dim in enumerate(VAD_DIMS)},
    )

    model_dir = Path(config.models.vad_dir)
    trainer = _run_trainer(
        config,
        model,
        tokenizer,
        _EncodedDataset(train_enc, train_labels),
        _EncodedDataset(val_enc, val_labels),
        model_dir,
        epochs,
        resume=resume,
    )

    test_preds: np.ndarray = np.asarray(
        trainer.predict(_EncodedDataset(test_enc, test_labels)).predictions
    )
    test_labels_np = test_labels.numpy()

    per_dim_ccc = {
        dim: _ccc(test_preds[:, i], test_labels_np[:, i]) for i, dim in enumerate(VAD_DIMS)
    }
    per_dim_pearson = {
        dim: float(np.corrcoef(test_preds[:, i], test_labels_np[:, i])[0, 1])
        for i, dim in enumerate(VAD_DIMS)
    }

    metrics = {
        "ccc": per_dim_ccc,
        "pearson_r": per_dim_pearson,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    return TrainResult("vad", model_dir, metrics)
