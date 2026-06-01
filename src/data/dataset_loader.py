from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_dataset_paths() -> dict:
    project_root = get_project_root()

    return {
        "project_root": project_root,
        "gtzan": project_root / "data" / "raw" / "gtzan" / "genres",
        "fma_medium": project_root / "data" / "raw" / "fma_medium",
        "fma_metadata": project_root / "data" / "raw" / "fma_metadata",
    }


def load_gtzan_metadata(gtzan_path: Path) -> pd.DataFrame:
    rows = []

    for genre_dir in sorted(gtzan_path.iterdir()):
        if genre_dir.is_dir():
            for audio_path in sorted(genre_dir.glob("*.au")):
                rows.append({
                    "dataset": "gtzan",
                    "filepath": str(audio_path),
                    "original_genre": genre_dir.name,
                    "label": genre_dir.name,
                })

    return pd.DataFrame(rows)


def load_fma_tracks(metadata_path: Path) -> pd.DataFrame:
    tracks_path = metadata_path / "tracks.csv"

    return pd.read_csv(
        tracks_path,
        header=[0, 1],
        index_col=0
    )


def select_fma_medium_shared_genres(
    tracks: pd.DataFrame,
    shared_genres: dict,
    fma_audio_path: Path
) -> pd.DataFrame:

    fma_medium = tracks[
        tracks[("set", "subset")] == "medium"
    ].copy()

    fma_medium = fma_medium[
        fma_medium[("track", "genre_top")].isin(shared_genres.keys())
    ].copy()

    rows = []

    for track_id, row in fma_medium.iterrows():

        original_genre = row[("track", "genre_top")]
        label = shared_genres[original_genre]

        folder = str(track_id).zfill(6)[:3]
        filename = str(track_id).zfill(6) + ".mp3"

        filepath = (
            fma_audio_path
            / folder
            / filename
        )

        rows.append({
            "dataset": "fma_medium",
            "track_id": track_id,
            "filepath": str(filepath),
            "original_genre": original_genre,
            "label": label,
            "duration": row[("track", "duration")],
        })

    return pd.DataFrame(rows)