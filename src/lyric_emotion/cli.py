"""Command-line entry points for the lyric-emotion pipeline.

Each command is implemented in its own phase; until then it raises
NotImplementedError so `--help` and the command surface are usable
from day one without pretending unfinished phases work.
"""

from typing import TYPE_CHECKING

import typer

from lyric_emotion.config import load_artists, load_config
from lyric_emotion.data import (
    fetch_all_artists,
    save_training_data,
    write_data_report,
    write_fetch_coverage_report,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lyric_emotion.model import TrainResult

app = typer.Typer(no_args_is_help=True)


def _write_train_report(result: "TrainResult", docs_dir: "Path | None" = None) -> None:
    import json
    from pathlib import Path

    docs_dir = docs_dir or Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / f"training_{result.head}.md"
    lines = [
        f"# Training report — {result.head} head",
        "",
        "```json",
        json.dumps(result.metrics, indent=2),
        "```",
    ]
    report_path.write_text("\n".join(lines))
    typer.echo(f"Wrote {report_path}")


@app.command()
def fetch() -> None:
    """Fetch lyrics for configured artists from Genius (Phase 3)."""
    from dotenv import load_dotenv

    load_dotenv()
    config = load_config()
    artists = load_artists().artists
    raw_artists = fetch_all_artists(config, artists)
    report_path = write_fetch_coverage_report(raw_artists)
    typer.echo(f"Wrote {report_path}")


@app.command(name="build-training")
def build_training() -> None:
    """Build redistributable training parquet files from GoEmotions/EmoBank (Phase 1)."""
    config = load_config()
    emotions_path, vad_path = save_training_data(config)
    report_path = write_data_report(emotions_path, vad_path)
    typer.echo(f"Wrote {emotions_path}")
    typer.echo(f"Wrote {vad_path}")
    typer.echo(f"Wrote {report_path}")


@app.command()
def train(
    head: str = typer.Option(..., help="'emotions' or 'vad'"),
    epochs: int = typer.Option(None, help="Override config epochs"),
    smoke: bool = typer.Option(
        False, help="Fast sanity run: subsampled data, short sequences, 1 epoch"
    ),
) -> None:
    """Fine-tune the emotion or VAD head on ModernBERT-base (Phase 2/4)."""
    from lyric_emotion.model import train_emotions, train_vad

    config = load_config()
    subsample = 500 if smoke else None
    run_epochs = 1 if smoke else epochs

    if head == "emotions":
        result = train_emotions(config, subsample=subsample, epochs=run_epochs)
    elif head == "vad":
        result = train_vad(config, subsample=subsample, epochs=run_epochs)
    else:
        raise typer.BadParameter("head must be 'emotions' or 'vad'")

    typer.echo(f"Saved model to {result.model_dir}")
    _write_train_report(result)


def _read_train_metrics(report_path: "Path") -> dict:
    import json

    block = report_path.read_text().split("```json\n")[1].split("\n```")[0]
    return json.loads(block)  # type: ignore[no-any-return]


@app.command(name="evaluate-baselines")
def evaluate_baselines() -> None:
    """Compare our fine-tuned heads against cirimus/modernbert-base-go-emotions
    and the NRC-VAD lexicon (Phase 2 acceptance criteria)."""
    from pathlib import Path

    from lyric_emotion.evaluate import (
        evaluate_emotions_baseline,
        evaluate_vad_baseline,
        write_baseline_report,
    )

    config = load_config()
    docs_dir = Path("docs")
    our_emotions = _read_train_metrics(docs_dir / "training_emotions.md")
    our_vad = _read_train_metrics(docs_dir / "training_vad.md")

    typer.echo("Scoring cirimus/modernbert-base-go-emotions on the validation split...")
    baseline_emotions = evaluate_emotions_baseline(config)
    typer.echo("Scoring the NRC-VAD lexicon on the EmoBank dev split...")
    baseline_vad = evaluate_vad_baseline(config)

    report_path = write_baseline_report(
        our_emotions, baseline_emotions, our_vad, baseline_vad, docs_dir
    )
    typer.echo(f"Wrote {report_path}")


@app.command()
def predict(text: str = typer.Option(..., help="Lyrics text to score")) -> None:
    """Run inference with the trained model (Phase 4/5)."""
    raise NotImplementedError("Phase 4/5: inference not implemented yet")


@app.command()
def analyze() -> None:
    """Aggregate scores by song/album/year and run statistics (Phase 5)."""
    raise NotImplementedError("Phase 5: analysis not implemented yet")


@app.command()
def report() -> None:
    """Generate final figures and tables (Phase 5/6)."""
    raise NotImplementedError("Phase 5/6: reporting not implemented yet")
