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


MEL_DIR = (
    PROCESSED_DIR
    / "mel"
)

MEL_TRAIN_PATH = (
    MEL_DIR
    / "mel_train.npy"
)

MEL_VALIDATION_PATH = (
    MEL_DIR
    / "mel_validation.npy"
)

MEL_TEST_PATH = (
    MEL_DIR
    / "mel_test.npy"
)

MEL_TRAIN_LABELS_PATH = (
    MEL_DIR
    / "mel_train_labels.npy"
)

MEL_VALIDATION_LABELS_PATH = (
    MEL_DIR
    / "mel_validation_labels.npy"
)

MEL_TEST_LABELS_PATH = (
    MEL_DIR
    / "mel_test_labels.npy"
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


CNN_OUTPUT_DIR = (
    OUTPUTS_DIR
    / "cnn"
)

CNN_MODEL_DIR = (
    MODELS_DIR
    / "cnn"
)

BEST_CNN_MODEL_PATH = (
    CNN_MODEL_DIR
    / "best_cnn_model.pt"
)

CNN_LABEL_ENCODER_PATH = (
    CNN_MODEL_DIR
    / "cnn_label_encoder.joblib"
)

CNN_METRICS_SUMMARY_PATH = (
    CNN_OUTPUT_DIR
    / "cnn_metrics_summary.csv"
)

CNN_TEST_CONFUSION_MATRIX_PATH = (
    CNN_OUTPUT_DIR
    / "cnn_test_confusion_matrix.png"
)

CNN_TRAINING_HISTORY_PATH = (
    CNN_OUTPUT_DIR
    / "cnn_training_history.csv"
)

CNN_TEST_REPORT_PATH = (
    CNN_OUTPUT_DIR
    / "cnn_test_classification_report.csv"
)


MOBILENET_OUTPUT_DIR = (
    OUTPUTS_DIR
    / "mobilenet"
)

MOBILENET_MODEL_DIR = (
    MODELS_DIR
    / "mobilenet"
)

BEST_MOBILENET_MODEL_PATH = (
    MOBILENET_MODEL_DIR
    / "best_mobilenet_model.pt"
)

MOBILENET_LABEL_ENCODER_PATH = (
    MOBILENET_MODEL_DIR
    / "mobilenet_label_encoder.joblib"
)

MOBILENET_METRICS_SUMMARY_PATH = (
    MOBILENET_OUTPUT_DIR
    / "mobilenet_metrics_summary.csv"
)

MOBILENET_TRAINING_HISTORY_PATH = (
    MOBILENET_OUTPUT_DIR
    / "mobilenet_training_history.csv"
)

MOBILENET_TEST_REPORT_PATH = (
    MOBILENET_OUTPUT_DIR
    / "mobilenet_test_classification_report.csv"
)

MOBILENET_TEST_CONFUSION_MATRIX_PATH = (
    MOBILENET_OUTPUT_DIR
    / "mobilenet_test_confusion_matrix.png"
)

QUANTIZED_MODEL_DIR = (
    MODELS_DIR
    / "quantized"
)

FP32_MOBILENET_DEPLOYMENT_PATH = (
    QUANTIZED_MODEL_DIR
    / "mobilenet_v2_fp32_deployment.pt"
)

STATIC_INT8_MOBILENET_MODEL_PATH = (
    QUANTIZED_MODEL_DIR
    / "mobilenet_v2_static_int8_quantized.pt"
)

DYNAMIC_LINEAR_MOBILENET_MODEL_PATH = (
    QUANTIZED_MODEL_DIR
    / "mobilenet_v2_dynamic_linear_quantized.pt"
)

CLASSIFIER_DYNAMIC_MOBILENET_MODEL_PATH = (
    QUANTIZED_MODEL_DIR
    / "mobilenet_v2_classifier_dynamic_quantized.pt"
)

QUANTIZATION_METRICS_SUMMARY_PATH = (
    MOBILENET_OUTPUT_DIR
    / "mobilenet_quantization_metrics_summary.csv"
)

QUANTIZATION_SIZE_COMPARISON_PATH = (
    MOBILENET_OUTPUT_DIR
    / "mobilenet_quantization_size_comparison.csv"
)

QUANTIZATION_SPEED_COMPARISON_PATH = (
    MOBILENET_OUTPUT_DIR
    / "mobilenet_quantization_speed_comparison.csv"
)

QUANTIZATION_CLASSIFICATION_REPORTS_PATH = (
    MOBILENET_OUTPUT_DIR
    / "mobilenet_quantization_classification_reports.csv"
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
# Mel Spectrograms and CNN
# ================================

N_MELS = 128

N_FFT = 2048

HOP_LENGTH = 512

MEL_TARGET_FRAMES = (
    int(
        SAMPLE_RATE
        * TARGET_DURATION
    )
    // HOP_LENGTH
    + 1
)

MEL_N_FFT = N_FFT

MEL_HOP_LENGTH = HOP_LENGTH

MEL_EXPECTED_FRAMES = MEL_TARGET_FRAMES

CNN_INPUT_SHAPE = (
    1,
    N_MELS,
    MEL_TARGET_FRAMES
)

CNN_BATCH_SIZE = 32

CNN_EPOCHS = 30

CNN_LEARNING_RATE = 0.001

CNN_RANDOM_STATE = 42

MOBILENET_BATCH_SIZE = 32

MOBILENET_EPOCHS = 15

MOBILENET_LEARNING_RATE = 0.0003


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
