from pathlib import Path


# ================================
# Project Paths
# ================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RAW_DIR = (
    DATA_DIR
    / "raw"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

SPLITS_DIR = (
    DATA_DIR
    / "splits"
)

TRAIN_DIR = (
    SPLITS_DIR
    / "train"
)

VALIDATION_DIR = (
    SPLITS_DIR
    / "validation"
)

TEST_DIR = (
    SPLITS_DIR
    / "test"
)


# ================================
# Metadata Files
# ================================

GTZAN_TRAIN_PATH = (
    TRAIN_DIR
    / "gtzan_train.csv"
)

GTZAN_VALIDATION_PATH = (
    VALIDATION_DIR
    / "gtzan_validation.csv"
)

FMA_EXTERNAL_PATH = (
    TEST_DIR
    / "fma_external.csv"
)


# ================================
# Audio Preprocessing
# ================================

SAMPLE_RATE = 22050

TARGET_DURATION = 28

MONO = True

CROP_MODE = "center"

NORMALIZATION = None


# ================================
# Shared Genres
# ================================

SHARED_GENRES = {

    "Blues": "blues",

    "Classical": "classical",

    "Country": "country",

    "Hip-Hop": "hiphop",

    "Jazz": "jazz",

    "Pop": "pop",

    "Rock": "rock",
}