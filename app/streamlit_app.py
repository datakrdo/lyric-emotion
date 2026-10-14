"""Dashboard: were The Cure and The Smiths equally gloomy?

Renders the bundle in `app/data/` written by `lyric-emotion export-app`.
It computes no statistics and holds no lyrics: every number and sentence
shown comes from `findings.json`.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#898781"

st.set_page_config(page_title="Equally gloomy?", layout="wide")


@st.cache_data(max_entries=1)
def load(bundle_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    # `bundle_mtime` only keys the cache: a regenerated bundle reloads by itself.
    songs = pd.read_parquet(DATA / "songs.parquet").reset_index(drop=True)
    songs["_i"] = songs.index
    return (
        songs,
        pd.read_csv(DATA / "albums.csv"),
        json.loads((DATA / "findings.json").read_text()),
        json.loads((DATA / "validation.json").read_text()),
    )


songs, albums, findings, validation = load(max(f.stat().st_mtime for f in DATA.iterdir()))
band_a, band_b = findings["bands"]
COLORS = {band_a: BLUE, band_b: ORANGE}
win = findings["window"]
span = f"{win['start_year']}–{win['end_year']}"
poles = findings["poles"]
STATUS_BADGE = {
    "clear": ("Clear", "green"),
    "suggestive": ("Suggestive", "orange"),
    "none": ("Not detected", "gray"),
}


def emotions_reading() -> str:
    """Which emotion families pass the strict album-level test (FDR-adjusted)."""
    fams = findings["families"]
    clear = [f["family"] for f in fams if f["status"] == "clear"]
    if clear:
        return (
            "Counting whole albums rather than songs, the difference holds up for "
            f"{', '.join(clear)}; not for the other families."
        )
    return (
        "Counting whole albums rather than songs (songs on one album sound alike), no single "
        "emotion family clearly separates the bands; some hint at it."
    )


def trust_rows() -> list[tuple[str, str, str, str]]:
    """(headline, badge text, badge color, plain-language reading)."""
    v, s, e = findings["valence"], findings["style"], findings["emotions"]
    lo, mid, hi = ((1 + v[k]) / 2 for k in ("ci_low", "delta", "ci_high"))
    if v["status"] == "none_detected":
        dark = (
            "No clear difference",
            "gray",
            f"Take one song from each band: the brighter one is {band_a}'s {mid:.0%} of the "
            f"time in this sample (plausible range {lo:.0%}–{hi:.0%}). A perfect tie is 50%, "
            "so a gap in either direction can't be ruled out.",
        )
    else:
        dark = ("Difference found", "blue", "The bands differ in how dark their lyrics read.")
    human = (
        ("Pending", "gray", "Not yet checked against ratings from a human reader.")
        if validation.get("status") == "pending"
        else ("Checked", "green", validation.get("summary", ""))
    )
    return [
        ("How dark", *dark),
        (
            "Overall feel",
            *STATUS_BADGE[s["status"]],
            f"Take one song from each band: its emotional profile says which band it's from "
            f"about {s['auc']:.0%} of the time (guessing gives 50%), tested on albums the "
            "model had not seen.",
        ),
        (
            "Single emotions",
            *STATUS_BADGE[e["status"]],
            emotions_reading(),
        ),
        ("Human check", *human),
    ]


def song_card(song: pd.Series) -> None:
    lean = band_a if song["style"] > 0 else band_b
    with st.container(border=True):
        st.markdown(
            f"**{song['title']}** · {song['artist']} · {song['album']} ({int(song['year'])})"
        )
        st.markdown(
            f"Brighter than **{song['valence_pct']:.0f}%** of the {len(songs)} studio songs · "
            f"overall feel leans **{lean}** · stands out for: {song['stands_out']}"
        )


def style_map() -> None:
    w = songs[songs["in_window"]]
    fig = px.scatter(
        w,
        x="valence_z",
        y="style",
        color="artist",
        color_discrete_map=COLORS,
        custom_data=["_i"],
        hover_name="title",
        hover_data={
            "album": True,
            "year": True,
            "valence_pct": ":.0f",
            "artist": False,
            "valence_z": False,
            "style": False,
            "_i": False,
        },
        labels={"valence_pct": "brighter than % of songs"},
        opacity=0.85,
        height=620,
    )
    fig.update_traces(
        marker={"size": 9, "line": {"width": 1.5, "color": "white"}}, selector={"type": "scatter"}
    )
    fig.update_layout(
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08, "title": None},
        font={"color": INK},
        margin={"t": 40},
    )
    fig.update_xaxes(
        showticklabels=False,
        zeroline=True,
        zerolinecolor="#e1e0d9",
        title="← darker              brighter →",
    )
    fig.update_yaxes(
        showticklabels=False,
        zeroline=True,
        zerolinecolor="#e1e0d9",
        title=f"↑ more {poles[band_a][0]}   ·   more {poles[band_b][0]} ↓",
    )
    for text, y, anchor, color in (
        (f"▲ more {', '.join(poles[band_a])}", 0.99, "top", BLUE),
        (f"▼ more {', '.join(poles[band_b])}", 0.01, "bottom", ORANGE),
    ):
        fig.add_annotation(
            text=text,
            xref="x domain",
            yref="y domain",
            x=0.01,
            y=y,
            xanchor="left",
            yanchor=anchor,
            showarrow=False,
            font={"color": color, "size": 13},
            row=1,
            col=1,
        )
    event = st.plotly_chart(
        fig, on_select="rerun", selection_mode="points", key="style_map", width="stretch"
    )
    picked = [p["customdata"][0] for p in event.selection.points if p.get("customdata")]
    if picked:
        song_card(songs.loc[int(picked[0])])
    else:
        st.caption("Each dot is one song. Click a dot to see where it sits.")


def answer() -> None:
    st.caption(f"LYRICS · {span} · {band_a.upper()} VS {band_b.upper()}")
    st.header(findings["verdict"], divider="gray")
    st.write(
        f"We had a language model read {win['n_songs']} songs from {win['n_studio_albums']} studio "
        "albums (plus singles and B-sides) and score the emotions in each. Below: where every "
        "song lands, and how sure we are."
    )
    style_map()

    st.subheader("How sure are we?")
    for headline, badge, color, reading in trust_rows():
        left, right = st.columns([1, 4], vertical_alignment="center")
        with left:
            st.markdown(f"**{headline}**")
            st.badge(badge, color=color)
        right.write(reading)

    st.subheader("What each band shows more of")
    compared = findings["emotions_compared"]
    rows = [
        {"emotion": r["emotion"] + (" (no difference)" if not r["significant"] else ""),
         "band": band, "score": 100 * r[key]}
        for r in compared
        for band, key in ((band_a, "median_a"), (band_b, "median_b"))
    ]
    fig = px.bar(
        rows, x="score", y="emotion", color="band", barmode="group", orientation="h",
        color_discrete_map=COLORS, height=80 + 70 * len(compared),
    )
    fig.update_layout(
        template="plotly_white", font={"color": INK}, legend={"orientation": "h", "title": None},
        yaxis={"autorange": "reversed", "title": None}, xaxis={"title": "Typical song, 0–100"},
        margin={"t": 10},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each bar is a band's typical song: how strongly the model detects that emotion "
        "(0–100). A song can be sad and joyful at once. Apart from sadness, every difference "
        "shown holds up even counting whole albums."
    )
    q = findings["question_line_share"]
    st.caption(
        f"A fact you can check: {q[band_b]:.1%} of {band_b}'s lines end with “?”, "
        f"against {q[band_a]:.1%} for {band_a}."
    )


def album_cards(band: str, show_other: bool) -> None:
    in_window = albums["year"].between(win["start_year"], win["end_year"])
    a = albums[(albums["artist"] == band) & in_window]
    a = (a if show_other else a[a["kind"] == "studio"]).sort_values("year")
    for _, album in a.iterrows():
        with st.container(border=True):
            cover, text = st.columns([1, 3], vertical_alignment="center")
            if isinstance(album["cover_url"], str) and album["cover_url"]:
                cover.image(album["cover_url"], width="stretch")
            text.markdown(f"**{album['album']}** ({int(album['year'])}) · {album['n']} songs")
            text.progress(album["valence_pct"] / 100, text="darker ← → brighter than other albums")
            text.caption(f"Stands out for: {album['stands_out']}")


def explore() -> None:
    st.title("Explore")
    show_other = st.checkbox("Include singles and compilations", value=False)
    st.subheader("Album by album")
    for col, name in zip(st.columns(2), (band_a, band_b), strict=True):
        with col:
            st.markdown(f":{'blue' if name == band_a else 'orange'}[**{name}**]")
            album_cards(name, show_other)

    st.subheader("Band career and songs")
    band = st.segmented_control("Band", [band_a, band_b], default=band_a, required=True)
    with st.expander("Brightness album by album, over the years"):
        a = albums[albums["artist"] == band]
        a = (a if show_other else a[a["kind"] == "studio"]).sort_values("year")
        a = a.assign(year=a["year"].astype(int))
        whole = st.checkbox("Show the whole career", value=False)
        first = int(a["year"].min())
        if not whole:  # default view: the band's first decade
            a = a[a["year"] <= first + 9]
        fig = go.Figure(
            go.Scatter(
                x=a["year"],
                y=a["valence_z_mean"],
                mode="lines+markers",
                line={"color": COLORS[band], "width": 2},
                marker={"size": 9, "color": COLORS[band]},
                error_y={
                    "type": "data",
                    "array": a["valence_z_ci_high"] - a["valence_z_mean"],
                    "arrayminus": a["valence_z_mean"] - a["valence_z_ci_low"],
                    "color": MUTED,
                    "thickness": 1,
                },
                text=a["album"],
                customdata=a["n"],
                hovertemplate="<b>%{text}</b> (%{x})<br>%{customdata} songs<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_color="#e1e0d9")
        fig.update_layout(
            template="plotly_white",
            height=420,
            font={"color": INK},
            yaxis={"title": "← darker      brighter →", "showticklabels": False},
            xaxis={"title": "Album release year", "dtick": 1, "tickformat": "d"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Dots are album averages; bars are 95% margins of error. Few songs per album "
            "means wide bars — read the shape, not single albums."
        )

    st.subheader("Find a song")
    query = st.text_input("Search by title")
    found = songs[songs["artist"] == band]
    if query:
        found = found[found["title"].str.contains(query, case=False, regex=False)]
    st.dataframe(
        found[["title", "album", "year", "valence_pct", "stands_out"]].rename(
            columns={"valence_pct": "brighter than %", "stands_out": "stands out for"}
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "brighter than %": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f"
            )
        },
    )


def how_it_works() -> None:
    st.title("How it works")
    st.markdown(
        f"""
**1 · Data.** Lyrics of studio songs by {band_a} and {band_b}, {span} for the comparison:
the years in which both were active.
Covers, live takes, demos and other artists' versions are filtered out. No lyrics are stored
or shown in this app, only scores.

**2 · Model.** A language model (ModernBERT) trained on public sets of short texts scores 28
emotions per song and a brightness (valence) score. Lyrics are not tweets, so a second stage
adapts the brightness score to song lyrics using weak, tag-based labels.

**3 · Checks.** Songs on one album sound alike, so we count *albums*, not songs, when testing
differences, and we test on albums the model has not seen. “Overall feel” is the best
combination of emotion scores for telling the bands apart; it is measured, not assumed.

**4 · Limits.** Lyrics only (no melody, voice or tempo); English only; irony and fictional
narrators can fool the model; scores are for comparing songs with each other, not absolute
sadness. We also tested *arousal* and *dominance* and did not report them: they didn't hold up.
"""
    )
    with st.expander("The numbers behind the verdict"):
        v, s = findings["valence"], findings["style"]
        st.write(
            f"Brightness gap ({band_a} − {band_b}), Cliff's delta {v['delta']:.2f}, 95% interval "
            f"[{v['ci_low']:.2f}, {v['ci_high']:.2f}] (albums resampled). Overall feel: "
            f"cross-validated AUC {s['auc']:.2f}; {s['n_permutations']} album-level shuffles, "
            f"p = {s['p_value']:.3f}; AUC without question-related emotions "
            f"{s['auc_without_question_labels']:.2f}."
        )
        for r in findings["emotions_compared"]:
            st.write(
                f"{r['emotion']}: {band_a}'s song scores higher in {r['pair_share_a']:.0%} of "
                f"song pairs; album-level p = {r['p_adj']:.3f} (corrected for 28 emotions)."
            )
        st.json(findings["meta"])
    with st.expander("Data sources and licenses"):
        st.write(
            "GoEmotions (Apache-2.0), EmoBank (CC BY-SA 4.0), MoodyLyrics and the Deezer "
            "Mood Detection Dataset (weak labels), NRC-VAD lexicon (baseline). Lyrics "
            "come from Genius and are not redistributed; album covers are linked, not hosted."
        )


st.navigation(
    [
        st.Page(answer, title="The answer", default=True),
        st.Page(explore, title="Explore"),
        st.Page(how_it_works, title="How it works"),
    ]
).run()
