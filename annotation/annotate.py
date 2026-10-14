"""Local-only annotation page for the gold set. Shows one lyric at a time with
the artist, title and model scores hidden; lyrics never leave this machine.

    uv run streamlit run annotation/annotate.py
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from lyric_emotion.config import load_config
from lyric_emotion.data import _slugify

GOLD = Path("annotation/gold_set.csv")
config = load_config()
FAMILIES = [*config.emotion_families, "Neutral / unclear"]

st.set_page_config(page_title="Gold set annotation", layout="centered")
gold = pd.read_csv(GOLD)
gold[["families", "notes", "annotation_date"]] = gold[
    ["families", "notes", "annotation_date"]
].astype("object")
corpus = pd.read_parquet(Path(config.paths.interim_dir) / "studio_corpus.parquet")
corpus["song_id"] = corpus["artist"].map(_slugify) + "__" + corpus["title"].map(_slugify)

redo = st.sidebar.checkbox("Re-annotation pass (songs already rated)")
if redo:
    rated = gold[(gold["is_reannotation"] == 0) & gold["valence"].notna()]
    done_again = set(gold.loc[gold["is_reannotation"] == 1, "song_id"])
    todo = rated[~rated["song_id"].isin(done_again)].sample(frac=1, random_state=7)
else:
    todo = gold[(gold["is_reannotation"] == 0) & gold["valence"].isna()]
rated_n = int(gold.loc[gold["is_reannotation"] == 0, "valence"].notna().sum())
st.sidebar.progress(rated_n / max(1, int((gold["is_reannotation"] == 0).sum())), f"{rated_n} rated")

if todo.empty:
    st.success("Nothing left in this pass.")
    st.stop()

row = todo.iloc[0]
lyrics = corpus.loc[corpus["song_id"] == row["song_id"], "lyrics"].iloc[0]
st.text(lyrics)

with st.form("rate", clear_on_submit=True):
    valence = st.slider("Valence (1 = very negative, 5 = very positive)", 1, 5, 3)
    arousal = st.slider("Arousal (1 = very calm, 5 = very intense)", 1, 5, 3)
    families = st.multiselect("Emotion families (at most 2)", FAMILIES, max_selections=2)
    low_conf = st.checkbox("Low confidence (sparse or ambiguous text)")
    notes = st.text_input("Notes (irony, narrator...)")
    if st.form_submit_button("Save and next"):
        entry = {
            "valence": valence,
            "arousal": arousal,
            "families": ";".join(families),
            "low_confidence": int(low_conf),
            "notes": notes,
            "annotation_date": date.today().isoformat(),
        }
        if redo:
            gold = pd.concat(
                [gold, pd.DataFrame([{**row.to_dict(), **entry, "is_reannotation": 1}])],
                ignore_index=True,
            )
        else:
            for k, v in entry.items():
                gold.loc[row.name, k] = v
        gold.to_csv(GOLD, index=False)
        st.rerun()
