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


MFCC_DIR = (
    PROCESSED_DIR
    / "mfcc"
)

MFCC_TRAIN_PATH = (
    MFCC_DIR
    / "mfcc_train.csv"
)

MFCC_VALIDATION_PATH = (
    MFCC_DIR
    / "mfcc_validation.csv"
)

MFCC_TEST_PATH = (
    MFCC_DIR
    / "mfcc_test.csv"
)

# ================================
# Model and Output Artifacts
# ================================

OUTPUTS_DIR = (
    PROJECT_ROOT
    / "outputs"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

CLASSICAL_ML_OUTPUT_DIR = (
    OUTPUTS_DIR
    / "classical_ml"
)

CLASSICAL_ML_MODEL_DIR = (
    MODELS_DIR
    / "classical_ml"
)

BEST_CLASSICAL_MODEL_PATH = (
    CLASSICAL_ML_MODEL_DIR
    / "best_classical_model.joblib"
)

CLASSICAL_LABEL_ENCODER_PATH = (
    CLASSICAL_ML_MODEL_DIR
    / "label_encoder.joblib"
)

CLASSICAL_SCALER_PATH = (
    CLASSICAL_ML_MODEL_DIR
    / "best_model_scaler.joblib"
)

CLASSICAL_METRICS_SUMMARY_PATH = (
    CLASSICAL_ML_OUTPUT_DIR
    / "classical_ml_metrics_summary.csv"
)

CLASSICAL_TEST_CONFUSION_MATRIX_PATH = (
    CLASSICAL_ML_OUTPUT_DIR
    / "test_confusion_matrix.png"
)

CLASSICAL_VALIDATION_REPORTS_PATH = (
    CLASSICAL_ML_OUTPUT_DIR
    / "validation_classification_reports.csv"
)

CLASSICAL_TEST_REPORT_PATH = (
    CLASSICAL_ML_OUTPUT_DIR
    / "test_classification_report.csv"
)


# ================================
# Metadata Files
# ================================

TRAIN_METADATA_PATH = (
    TRAIN_DIR
    / "train_metadata.csv"
)

VALIDATION_METADATA_PATH = (
    VALIDATION_DIR
    / "validation_metadata.csv"
)

TEST_METADATA_PATH = (
    TEST_DIR
    / "test_metadata.csv"
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

SHARED_LABELS = list(
    SHARED_GENRES.values()
)
