from lyric_emotion.config import load_artists, load_config


def test_load_config_has_no_hardcoded_gaps():
    config = load_config()
    assert config.project.random_seed == 42
    assert config.training.base_model == "answerdotai/ModernBERT-base"
    assert config.emotion_head.num_labels == 28
    assert config.aggregation.min_songs_per_album >= 1


def test_load_artists_has_the_case_study_bands():
    artists = load_artists()
    names = {a.name for a in artists.artists}
    assert {"The Cure", "The Smiths"} <= names
    assert all(a.max_songs > 0 for a in artists.artists)
