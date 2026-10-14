"""Phase 5 figures — the plan's four required charts. Plain matplotlib
static PNGs (dpi from config), categorical palette: blue for The Cure,
orange for The Smiths, single-hue blue ramp for magnitude (heatmap).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from lyric_emotion.config import Visualization
from lyric_emotion.data import GOEMOTIONS_LABELS

_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"
ARTIST_COLORS = {"The Cure": "#2a78d6", "The Smiths": "#eb6834"}
_SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#2a78d6", "#0d366b"])


def _style_axes(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(_MUTED)
    ax.spines["bottom"].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)
    ax.grid(color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def va_trajectory_figure(album_scores: pd.DataFrame, viz: Visualization, path: Path) -> Path:
    """Signature figure: each artist's albums in the valence-arousal plane,
    connected with chronological arrows, marker size ~ songs per album.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for artist, group in album_scores.groupby("artist"):
        group = group.sort_values("year")
        color = ARTIST_COLORS.get(artist, _MUTED)
        xs, ys = group["valence_mean"].to_numpy(), group["arousal_mean"].to_numpy()
        for i in range(len(group) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.5, "lw": 1},
            )
        ax.scatter(
            xs,
            ys,
            s=group["n"] * 6,
            color=color,
            label=artist,
            zorder=2,
            edgecolor="white",
            linewidth=0.5,
        )
        for label_row in (group.iloc[0], group.iloc[-1]):
            ax.annotate(
                f"{label_row['album']} ({int(label_row['year'])})",
                (label_row["valence_mean"], label_row["arousal_mean"]),
                fontsize=7,
                color=_INK,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.axhline(0, color=_GRID, linewidth=0.8)
    ax.axvline(0, color=_GRID, linewidth=0.8)
    ax.set_xlabel("valence")
    ax.set_ylabel("arousal")
    ax.set_title("Album trajectories in the valence-arousal plane")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=viz.dpi, facecolor=_SURFACE)
    plt.close(fig)
    return path


def emotion_heatmap_figure(by_album: pd.DataFrame, viz: Visualization, path: Path) -> Path:
    """Album x 28-emotion heatmap, one hue (blue), light->dark = more."""
    prob_cols = [f"prob_{label}_mean" for label in GOEMOTIONS_LABELS]
    rows = by_album["artist"] + " — " + by_album["album"]
    matrix = by_album[prob_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(by_album))))
    im = ax.imshow(matrix, cmap=_SEQUENTIAL_BLUE, aspect="auto", vmin=0, vmax=matrix.max())
    ax.set_xticks(range(len(GOEMOTIONS_LABELS)))
    ax.set_xticklabels(GOEMOTIONS_LABELS, rotation=90, fontsize=7, color=_INK)
    ax.set_yticks(range(len(by_album)))
    ax.set_yticklabels(rows, fontsize=7, color=_INK)
    ax.set_title("Mean emotion probability by album")
    fig.colorbar(im, ax=ax, label="mean probability", shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=viz.dpi, facecolor=_SURFACE)
    plt.close(fig)
    return path


def valence_timeseries_figure(by_year: pd.DataFrame, viz: Visualization, path: Path) -> Path:
    """Valence over time per artist, with bootstrap CI band and n labeled at
    each point — never hide the sample size behind a smooth line.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for artist, group in by_year.groupby("artist"):
        group = group.sort_values("year")
        color = ARTIST_COLORS.get(artist, _MUTED)
        ax.plot(group["year"], group["valence_mean"], color=color, label=artist, marker="o")
        ax.fill_between(
            group["year"],
            group["valence_ci_low"],
            group["valence_ci_high"],
            color=color,
            alpha=0.15,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"n={int(row['n'])}",
                (row["year"], row["valence_ci_high"]),
                fontsize=6,
                color=_MUTED,
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
    ax.axhline(0, color=_GRID, linewidth=0.8)
    ax.set_xlabel("year")
    ax.set_ylabel("valence")
    ax.set_title("Valence by year (mean, bootstrap 95% CI, n per point)")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=viz.dpi, facecolor=_SURFACE)
    plt.close(fig)
    return path


def emotion_distribution_figure(song_scores: pd.DataFrame, viz: Visualization, path: Path) -> Path:
    """Mean per-emotion probability, one grouped horizontal bar per artist,
    sorted by overall mean so the most common emotions read top-down.
    """
    prob_cols = [f"prob_{label}" for label in GOEMOTIONS_LABELS]
    means = song_scores.groupby("artist")[prob_cols].mean()
    order = means.mean(axis=0).sort_values(ascending=True).index
    labels = [c.removeprefix("prob_") for c in order]

    fig, ax = plt.subplots(figsize=(8, 10))
    bar_height = 0.35
    y = range(len(order))
    for i, (artist, row) in enumerate(means.iterrows()):
        offset = (i - (len(means) - 1) / 2) * bar_height
        ax.barh(
            [v + offset for v in y],
            row[order],
            height=bar_height,
            color=ARTIST_COLORS.get(artist, _MUTED),
            label=artist,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8, color=_INK)
    ax.set_xlabel("mean probability")
    ax.set_title("Emotion distribution by artist")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=viz.dpi, facecolor=_SURFACE)
    plt.close(fig)
    return path


def write_all_figures(
    album_scores: pd.DataFrame,
    by_album: pd.DataFrame,
    by_year: pd.DataFrame,
    song_scores: pd.DataFrame,
    viz: Visualization,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        va_trajectory_figure(album_scores, viz, output_dir / "va_trajectory.png"),
        emotion_heatmap_figure(by_album, viz, output_dir / "emotion_heatmap.png"),
        valence_timeseries_figure(by_year, viz, output_dir / "valence_timeseries.png"),
        emotion_distribution_figure(song_scores, viz, output_dir / "emotion_distribution.png"),
    ]
