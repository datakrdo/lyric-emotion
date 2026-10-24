"""Command-line entry points for the lyric-emotion pipeline."""

from typing import TYPE_CHECKING

import pandas as pd
import typer

from lyric_emotion.config import load_artists, load_config
from lyric_emotion.data import (
    GOEMOTIONS_LABELS,
    fetch_all_artists,
    save_studio_corpus,
    save_training_data,
    write_data_report,
    write_dedup_report,
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
    """Fetch lyrics for configured artists from Genius, then dedupe live/demo/
    remix/reissue versions and write the studio-only corpus (Phase 3)."""
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    config = load_config()
    artists = load_artists().artists
    raw_artists = fetch_all_artists(config, artists)

    coverage_path = write_fetch_coverage_report(raw_artists)
    typer.echo(f"Wrote {coverage_path}")

    dedup_path = write_dedup_report(raw_artists, artists=artists)
    typer.echo(f"Wrote {dedup_path}")

    corpus_path = save_studio_corpus(
        raw_artists, Path(config.paths.interim_dir), comparison=config.comparison, artists=artists
    )
    typer.echo(f"Wrote {corpus_path}")
    typer.echo(f"Wrote {Path(config.paths.interim_dir) / 'song_albums.parquet'}")
    typer.echo(f"Wrote {Path(config.paths.interim_dir) / 'studio_corpus_comparison.parquet'}")


@app.command(name="build-training")
def build_training() -> None:
    """Build redistributable training parquet files from GoEmotions/EmoBank (Phase 1)."""
    config = load_config()
    emotions_path, vad_path = save_training_data(config)
    report_path = write_data_report(emotions_path, vad_path)
    typer.echo(f"Wrote {emotions_path}")
    typer.echo(f"Wrote {vad_path}")
    typer.echo(f"Wrote {report_path}")


@app.command(name="build-domain-training")
def build_domain_training() -> None:
    """Fetch MoodyLyrics + a sample of DMDD lyrics from Genius and build the
    weak-labeled domain-adaptation set for the Phase 4 stage-2 fine-tune."""
    from pathlib import Path

    from dotenv import load_dotenv

    from lyric_emotion.data import build_domain_dataset, write_domain_data_report

    load_dotenv()
    config = load_config()
    df = build_domain_dataset(config)

    out_path = Path(config.paths.processed_dir) / "domain_train.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    typer.echo(f"Wrote {out_path}")

    report_path = write_domain_data_report(df)
    typer.echo(f"Wrote {report_path}")


@app.command()
def train(
    head: str = typer.Option(..., help="'emotions', 'vad', or 'vad-domain'"),
    epochs: int = typer.Option(None, help="Override config epochs"),
    smoke: bool = typer.Option(
        False, help="Fast sanity run: subsampled data, short sequences, 1 epoch"
    ),
    resume: bool = typer.Option(False, help="Resume from the last saved checkpoint"),
    sources: str = typer.Option(None, help="vad-domain only: comma-separated sources, e.g. dmdd"),
    seed: int = typer.Option(None, help="vad-domain only: override the random seed"),
    out: str = typer.Option(None, help="vad-domain only: output model directory"),
) -> None:
    """Fine-tune the emotion or VAD head on ModernBERT-base (Phase 2/4)."""
    import dataclasses
    from pathlib import Path

    from lyric_emotion.model import train_emotions, train_vad, train_vad_domain

    config = load_config()
    subsample = 500 if smoke else None
    run_epochs = 1 if smoke else epochs

    if head == "emotions":
        result = train_emotions(config, subsample=subsample, epochs=run_epochs, resume=resume)
    elif head == "vad":
        result = train_vad(config, subsample=subsample, epochs=run_epochs, resume=resume)
    elif head == "vad-domain":
        result = train_vad_domain(
            config,
            subsample=subsample,
            epochs=run_epochs,
            resume=resume,
            sources=tuple(sources.split(",")) if sources else None,
            seed=seed,
            model_dir=Path(out) if out else None,
        )
        if out:  # ablation run: keep its report apart from the main one
            result = dataclasses.replace(result, head=Path(out).name)
    else:
        raise typer.BadParameter("head must be 'emotions', 'vad', or 'vad-domain'")

    typer.echo(f"Saved model to {result.model_dir}")
    _write_train_report(result)


@app.command(name="evaluate-baselines")
def evaluate_baselines() -> None:
    """Compare our fine-tuned heads against cirimus/modernbert-base-go-emotions
    and the NRC-VAD lexicon, and stage-1 vs stage-2 VAD checkpoints against
    each other (Phase 2/4 acceptance criteria)."""
    from pathlib import Path

    from lyric_emotion.evaluate import (
        emotion_vad_consistency,
        evaluate_emotions_baseline,
        evaluate_vad_baseline,
        evaluate_vad_checkpoints,
        write_baseline_report,
        write_consistency_report,
        write_vad_checkpoint_report,
    )
    from lyric_emotion.model import read_train_metrics

    config = load_config()
    docs_dir = Path("docs")
    our_emotions = read_train_metrics(docs_dir / "training_emotions.md")
    our_vad = read_train_metrics(docs_dir / "training_vad.md")

    typer.echo("Scoring cirimus/modernbert-base-go-emotions on the test split...")
    baseline_emotions = evaluate_emotions_baseline(config)
    typer.echo("Scoring the NRC-VAD lexicon on the EmoBank test split...")
    baseline_vad = evaluate_vad_baseline(config)

    report_path = write_baseline_report(
        our_emotions, baseline_emotions, our_vad, baseline_vad, docs_dir
    )
    typer.echo(f"Wrote {report_path}")

    typer.echo("Comparing vad vs vad_domain checkpoints on the DMDD test split...")
    checkpoint_result = evaluate_vad_checkpoints(config)
    checkpoint_path = write_vad_checkpoint_report(checkpoint_result, docs_dir)
    typer.echo(f"Wrote {checkpoint_path}")

    song_emotions_path = Path(config.paths.processed_dir) / "song_emotions.parquet"
    if song_emotions_path.exists():
        song_scores = pd.read_parquet(song_emotions_path)
        consistency = emotion_vad_consistency(config, song_scores)
        consistency_path = write_consistency_report(consistency, docs_dir)
        typer.echo(f"Wrote {consistency_path}")


@app.command(name="gold-sample")
def gold_sample(n: int = typer.Option(80, help="Songs to annotate")) -> None:
    """Draw the light gold set (stratified, model scores hidden) into annotation/gold_set.csv."""
    from pathlib import Path

    from lyric_emotion.gold import sample_light_gold_set

    config = load_config()
    path = Path("annotation/gold_set.csv")
    if path.exists() and pd.read_csv(path)["valence"].notna().any():
        raise typer.BadParameter(f"{path} already has annotations; move it first")
    window = pd.read_parquet(Path(config.paths.interim_dir) / "studio_corpus_comparison.parquet")
    scores = pd.read_parquet(Path(config.paths.processed_dir) / "song_emotions.parquet")
    merged = window[["artist", "title"]].merge(scores, on=["artist", "title"])
    sample = sample_light_gold_set(merged, n, config.project.random_seed)
    sample.to_csv(path, index=False)
    typer.echo(
        f"Wrote {path} ({len(sample)} songs). Annotate with: streamlit run annotation/annotate.py"
    )


@app.command(name="evaluate-gold")
def evaluate_gold_cmd() -> None:
    """Compare model scores with the human gold set; writes docs/gold_evaluation.md
    and the dashboard's validation.json (Phase 4)."""
    import json
    from pathlib import Path

    from lyric_emotion.data import _slugify
    from lyric_emotion.gold import evaluate_gold_set, validation_summary

    config = load_config()
    gold = pd.read_csv(Path("annotation/gold_set.csv"))
    scores = pd.read_parquet(Path(config.paths.processed_dir) / "song_emotions.parquet")
    scores["song_id"] = scores["artist"].map(_slugify) + "__" + scores["title"].map(_slugify)
    scores = scores.rename(columns={"valence": "valence_model", "arousal": "arousal_model"})
    result = evaluate_gold_set(gold, scores, config.emotion_families, config.project.random_seed)
    result["status"] = "done"
    result["summary"] = validation_summary(result)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/gold_evaluation.md").write_text(
        "# Gold-set evaluation\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n"
    )
    out = Path("site/public/data/validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    typer.echo(f"{result['summary']}\nWrote docs/gold_evaluation.md and {out}")


@app.command()
def predict(text: str = typer.Option(..., help="Lyrics text to score")) -> None:
    """Score one piece of text with both trained heads (Phase 4/5)."""
    import json

    from lyric_emotion.model import VAD_DIMS, predict_texts

    config = load_config()
    row = predict_texts(config, [text]).iloc[0]

    active_emotions = {
        label: round(float(row[f"prob_{label}"]), 3)
        for label in GOEMOTIONS_LABELS
        if row[label] == 1
    }
    result = {
        "emotions": active_emotions,
        "vad": {dim: round(float(row[dim]), 3) for dim in VAD_DIMS},
    }
    typer.echo(json.dumps(result, indent=2))


@app.command()
def analyze() -> None:
    """Score the studio corpus with both heads, then aggregate by album/year
    and estimate per-artist temporal trends (Phase 5)."""
    from pathlib import Path

    from lyric_emotion.analysis import aggregate_by_album, aggregate_by_year, temporal_regression
    from lyric_emotion.data import GOEMOTIONS_LABELS
    from lyric_emotion.model import VAD_DIMS, predict_texts

    config = load_config()
    interim_dir = Path(config.paths.interim_dir)
    output_dir = Path(config.paths.output_dir) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(interim_dir / "studio_corpus.parquet")
    scores = predict_texts(config, corpus["lyrics"].tolist()).drop(columns="text")
    song_scores = pd.concat(
        [corpus[["artist", "title", "album", "release_date"]].reset_index(drop=True), scores],
        axis=1,
    )
    song_scores["year"] = pd.to_numeric(song_scores["release_date"].str.slice(0, 4))

    from lyric_emotion.data import assign_album_groups, label_studio_albums

    album_map = label_studio_albums(
        pd.read_parquet(interim_dir / "song_albums.parquet"), load_artists().artists
    )
    song_scores = assign_album_groups(song_scores, album_map)

    song_emotions_path = Path(config.paths.processed_dir) / "song_emotions.parquet"
    song_scores.to_parquet(song_emotions_path, index=False)
    typer.echo(f"Wrote {song_emotions_path}")

    score_cols = [f"prob_{label}" for label in GOEMOTIONS_LABELS] + VAD_DIMS

    album_scores = album_map.merge(
        song_scores.drop(columns=["album", "release_date", "album_group"]), on=["artist", "title"]
    )
    seed = config.project.random_seed
    by_album = aggregate_by_album(
        album_scores, score_cols, config.aggregation, config.statistics, seed
    )
    from lyric_emotion.analysis import mode_year

    album_years = album_scores.groupby(["artist", "album"])["year"].agg(mode_year).reset_index()
    by_album = by_album.merge(album_years, on=["artist", "album"])
    by_album.to_csv(output_dir / "by_album.csv", index=False)
    typer.echo(f"Wrote {output_dir / 'by_album.csv'}")

    by_year = aggregate_by_year(
        song_scores, score_cols, config.aggregation, config.statistics, seed
    )
    by_year.to_csv(output_dir / "by_year.csv", index=False)
    typer.echo(f"Wrote {output_dir / 'by_year.csv'}")

    trends = pd.concat(
        [temporal_regression(song_scores, dim).assign(dim=dim) for dim in VAD_DIMS],
        ignore_index=True,
    )
    trends.to_csv(output_dir / "temporal_trends.csv", index=False)
    typer.echo(f"Wrote {output_dir / 'temporal_trends.csv'}")


@app.command()
def report() -> None:
    """Compare artists over the overlap window and generate the four Phase 5
    figures (Phase 5/6)."""
    from pathlib import Path

    from lyric_emotion.analysis import compare_artists
    from lyric_emotion.data import GOEMOTIONS_LABELS
    from lyric_emotion.figures import write_all_figures
    from lyric_emotion.model import VAD_DIMS

    config = load_config()
    interim_dir = Path(config.paths.interim_dir)
    processed_dir = Path(config.paths.processed_dir)
    tables_dir = Path(config.paths.output_dir) / "tables"

    song_scores = pd.read_parquet(processed_dir / "song_emotions.parquet")
    by_album = pd.read_csv(tables_dir / "by_album.csv")
    by_year = pd.read_csv(tables_dir / "by_year.csv")

    comparison_titles = pd.read_parquet(interim_dir / "studio_corpus_comparison.parquet")[
        ["artist", "title"]
    ]
    comparison_scores = comparison_titles.merge(song_scores, on=["artist", "title"])
    score_cols = [f"prob_{label}" for label in GOEMOTIONS_LABELS] + VAD_DIMS
    band_a, band_b = (a.name for a in load_artists().artists[:2])
    albums = comparison_scores["album_group"]
    comparison = compare_artists(
        comparison_scores,
        score_cols,
        band_a,
        band_b,
        config.statistics,
        seed=config.project.random_seed,
        groups=albums,
        permutations=config.statistics.permutations,
    )
    comparison_path = tables_dir / "artist_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    typer.echo(f"Wrote {comparison_path}")

    studio_albums = by_album[by_album["kind"] == "studio"] if "kind" in by_album else by_album
    figure_paths = write_all_figures(
        studio_albums,
        studio_albums,
        by_year,
        song_scores,
        config.visualization,
        Path(config.paths.output_dir) / "figures",
    )
    for figure_path in figure_paths:
        typer.echo(f"Wrote {figure_path}")


@app.command(name="export-app")
def export_app(out: str = typer.Option("site/public/data", help="Bundle directory")) -> None:
    """Write the dashboard's data bundle (songs, albums, findings; no lyrics)."""
    import json
    import re
    from pathlib import Path

    from lyric_emotion.data import _slugify, canonical_studio_album
    from lyric_emotion.export import build_bundle, question_line_share

    config = load_config()
    bands = tuple(a.name for a in load_artists().artists[:2])
    processed_dir = Path(config.paths.processed_dir)
    scores = pd.read_parquet(processed_dir / "song_emotions.parquet")
    window = pd.read_parquet(Path(config.paths.interim_dir) / "studio_corpus_comparison.parquet")
    by_album = pd.read_csv(Path(config.paths.output_dir) / "tables" / "by_album.csv")

    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()  # noqa: E731
    domain_path = processed_dir / "domain_train.parquet"
    domain = (
        pd.read_parquet(domain_path, columns=["artist", "title", "split"])
        if domain_path.exists()
        else pd.DataFrame(columns=["artist", "title", "split"])
    )
    trained_on = {
        (norm(a), norm(t)) for a, t in zip(domain["artist"], domain["title"], strict=True)
    }
    domain_keys = {
        (a, t)
        for a, t in zip(scores["artist"], scores["title"], strict=True)
        if (norm(a), norm(t)) in trained_on
    }

    songs, albums, findings = build_bundle(
        config,
        scores,
        set(zip(window["artist"], window["title"], strict=True)),
        by_album,
        {b: question_line_share(window.loc[window["artist"] == b, "lyrics"]) for b in bands},
        domain_keys,
        bands,  # type: ignore[arg-type]
    )
    covers: dict[tuple[str, str], str] = {}
    for artist in load_artists().artists[:2]:
        raw_path = Path(config.paths.raw_dir) / "genius" / f"{_slugify(artist.name)}.json"
        if not raw_path.exists():
            continue
        for song in json.loads(raw_path.read_text())["songs"]:
            album = song.get("album") or {}
            name, url = album.get("name"), album.get("cover_art_url")
            if name and url:
                canonical = canonical_studio_album(name, artist.studio_albums) or name
                key = (artist.name, canonical)
                if name.casefold() == canonical.casefold() or key not in covers:
                    covers[key] = url  # the exact album name beats Deluxe/Anniversary/etc.
    albums["cover_url"] = [
        covers.get((a, b), "") for a, b in zip(albums["artist"], albums["album"], strict=True)
    ]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "findings": findings,
        "albums": albums.fillna("").to_dict("records"),
        "songs": songs[songs["in_window"]].drop(columns="in_window").to_dict("records"),
    }
    (out_dir / "bundle.json").write_text(json.dumps(bundle, sort_keys=True))
    validation = out_dir / "validation.json"
    if not validation.exists():
        validation.write_text(json.dumps({"status": "pending"}))
    typer.echo(f"Wrote {out_dir}/ (bundle.json, validation.json)")
