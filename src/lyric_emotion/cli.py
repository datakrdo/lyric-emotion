"""Command-line entry points for the lyric-emotion pipeline.

Each command is implemented in its own phase; until then it raises
NotImplementedError so `--help` and the command surface are usable
from day one without pretending unfinished phases work.
"""

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def fetch() -> None:
    """Fetch lyrics for configured artists from Genius (Phase 3)."""
    raise NotImplementedError("Phase 3: lyrics acquisition not implemented yet")


@app.command(name="build-training")
def build_training() -> None:
    """Build redistributable training parquet files from GoEmotions/EmoBank (Phase 1)."""
    raise NotImplementedError("Phase 1: training data build not implemented yet")


@app.command()
def train(head: str = typer.Option(..., help="'emotions' or 'vad'")) -> None:
    """Fine-tune the emotion or VAD head on ModernBERT-base (Phase 2/4)."""
    raise NotImplementedError("Phase 2/4: training not implemented yet")


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
