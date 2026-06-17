# Methodology

## Overview

This project implements a complete music genre classification workflow using GTZAN and FMA Medium. The workflow is organized across seven notebooks and reusable modules in `src/`. The implementation first explores and harmonizes the datasets, then creates fixed train, validation, and test metadata splits, extracts MFCC and Mel Spectrogram features, trains classical and deep learning models, evaluates them with class-imbalance-aware metrics, and finally applies post-training quantization to the MobileNetV2 model.

## Dataset Exploration

Notebook 01 inspects both datasets before model development. GTZAN is explored through its genre folder structure and `.au` files. FMA Medium is explored through `tracks.csv`, including subset membership, top-level genre labels, and duration statistics. The notebook also introduces waveform, frequency spectrum, Mel Spectrogram, and MFCC visualizations to motivate the later feature extraction strategy.

The exploration shows that GTZAN contains short, standardized recordings, while FMA Medium contains longer recordings and a naturally imbalanced genre distribution. This motivates duration standardization, shared-label harmonization, stratified splitting, and the use of macro F1 in evaluation.

## Dataset Harmonization

Notebook 02 loads GTZAN metadata and FMA metadata through `src.data.dataset_loader`. GTZAN labels are obtained from folder names. FMA Medium rows are selected from `tracks.csv` where `("set", "subset") == "medium"`, and audio paths are constructed from the zero-padded FMA track identifiers.

No raw audio files are permanently modified during harmonization. The notebook creates metadata tables that point to the original downloaded files.

## Shared Genre Mapping

The final classification task uses seven labels shared between GTZAN and FMA Medium:

| FMA Genre | Project Label |
|---|---|
| Blues | `blues` |
| Classical | `classical` |
| Country | `country` |
| Hip-Hop | `hiphop` |
| Jazz | `jazz` |
| Pop | `pop` |
| Rock | `rock` |

GTZAN is filtered to the same seven labels. FMA Medium is filtered to these top-level genres and mapped through the same label names. The combined metadata table contains 9,445 rows before feature extraction skips: 700 GTZAN examples and 8,745 FMA Medium examples.

## Train / Validation / Test Split

Notebook 02 creates stratified splits from the combined metadata:

- 70 percent training
- approximately 15 percent validation
- 15 percent test

The implementation uses `train_test_split` twice with `random_state=42` and stratification by `label`. The initial metadata split sizes are:

| Split | Metadata Rows |
|---|---:|
| Train | 6,611 |
| Validation | 1,417 |
| Test | 1,417 |

The split metadata files are saved under `data/splits/` and reused by all later notebooks.

## Audio Preprocessing

Audio preprocessing is implemented in `src.data.preprocessing` and used by Notebooks 02 and 03. The configured parameters are defined in `src.utils.config`:

| Parameter | Value |
|---|---:|
| Sample rate | 22,050 Hz |
| Target duration | 28 seconds |
| Channels | Mono |
| Crop mode | Center crop |
| Normalization | None |

The preprocessing pipeline loads audio with `librosa`, checks that the recording is at least 28 seconds long, center-crops the signal to exactly 28 seconds, and optionally applies peak normalization. In the executed workflow, normalization is disabled to preserve musical dynamics. Recordings that cannot be decoded or do not satisfy preprocessing requirements are skipped during feature extraction.

## MFCC Feature Extraction

Notebook 03 extracts MFCC features for the classical machine learning workflow. The implementation computes 20 MFCC coefficients with `librosa.feature.mfcc`, then aggregates each coefficient over time using mean and standard deviation. This produces a fixed-length vector of 40 numerical features per audio file.

The generated CSV files are:

- `data/processed/mfcc/mfcc_train.csv`
- `data/processed/mfcc/mfcc_validation.csv`
- `data/processed/mfcc/mfcc_test.csv`

After skipped files, the MFCC datasets contain:

| Split | Rows | Columns |
|---|---:|---:|
| Train | 6,605 | 41 |
| Validation | 1,415 | 41 |
| Test | 1,414 | 41 |

The 41 columns consist of one `label` column and 40 `mfcc_*` feature columns.

## Mel Spectrogram Extraction

Notebook 03 also extracts Mel Spectrogram tensors for the CNN and MobileNetV2 workflows. The extraction uses:

- 128 Mel bands
- `n_fft=2048`
- `hop_length=512`
- target frame count of 1,206
- decibel conversion
- clipping to `[-80, 0]` dB and scaling to `[0, 1]`
- one channel dimension for PyTorch tensors

The generated arrays are:

- `data/processed/mel/mel_train.npy`
- `data/processed/mel/mel_validation.npy`
- `data/processed/mel/mel_test.npy`
- matching `*_labels.npy` files for each split

The executed tensor shapes are:

| Split | Shape | Skipped Files |
|---|---|---:|
| Train | `(6605, 1, 128, 1206)` | 6 |
| Validation | `(1415, 1, 128, 1206)` | 2 |
| Test | `(1414, 1, 128, 1206)` | 3 |

## Classical ML Workflow

Notebook 04 trains classical baselines on MFCC summary vectors. Labels are encoded with `LabelEncoder`. The model factories are implemented in `src.models`:

- SVM with `StandardScaler`, RBF kernel, `C=10.0`, `gamma="scale"`, and `class_weight="balanced"`.
- Random Forest with 400 trees, `class_weight="balanced"`, `random_state=42`, and `n_jobs=-1`.

Both models are trained on the train split and evaluated on train and validation splits. Validation macro F1 is used for model selection. The SVM is selected as the best classical model and is evaluated once on the test split. Artifacts are saved under `models/classical_ml/` and `outputs/classical_ml/`.

## CNN Baseline Workflow

Notebook 05 trains a small PyTorch CNN on saved Mel Spectrogram tensors. The architecture in `src.models.cnn_model` contains:

- three convolutional blocks,
- batch normalization,
- ReLU activations,
- max pooling in the first two blocks,
- adaptive average pooling,
- a dense classifier with dropout.

Training uses Adam, cross-entropy loss, mini-batches, validation-loss monitoring, early stopping with best-weight restoration, and the configured maximum of 30 epochs. The executed run stopped after 13 epochs. Artifacts are saved under `models/cnn/` and `outputs/cnn/`.

## MobileNetV2 Transfer Learning Workflow

Notebook 06 trains an ImageNet-pretrained MobileNetV2 classifier on the saved Mel Spectrogram tensors. The wrapper in `src.models.mobilenet_model` converts each single-channel Mel tensor to a three-channel input, resizes it to `224 x 224`, applies ImageNet normalization, and forwards it through MobileNetV2.

The pretrained convolutional feature extractor is frozen in the implemented experiment. Only the replaced classifier head is trained. The executed model has 8,967 trainable parameters out of 2,232,839 total parameters. Training uses Adam, cross-entropy loss, validation-loss early stopping, and the configured maximum of 15 epochs. Artifacts are saved under `models/mobilenet/` and `outputs/mobilenet/`.

## Quantization Workflow

Notebook 07 loads the saved MobileNetV2 model and evaluates deployment-oriented post-training quantization on CPU. It does not retrain models or regenerate features. The implemented quantization variants are:

- FP32 MobileNetV2 deployment artifact.
- Static INT8 quantization of the MobileNetV2 core using FX graph mode and validation-only calibration.
- Dynamic quantization of supported `Linear` layers.
- Classifier-only dynamic quantization.

The notebook saves TorchScript deployment artifacts where possible and compares predictive metrics, model size, throughput, and latency. The static INT8 variant reduces size and latency substantially but causes severe macro F1 degradation. Dynamic quantization and classifier-only dynamic quantization preserve predictive performance while providing smaller deployment benefits.

## Artifact Generation

The project generates the following artifact groups:

| Stage | Generated Artifacts |
|---|---|
| Notebook 02 | Train, validation, and test metadata CSV files under `data/splits/` |
| Notebook 03 | MFCC CSV files under `data/processed/mfcc/` and Mel tensors under `data/processed/mel/` |
| Notebook 04 | Best classical model, label encoder, optional scaler, metrics CSV files, classification reports, and confusion matrix |
| Notebook 05 | CNN model weights, label encoder, training history, metrics summary, test report, and confusion matrix |
| Notebook 06 | MobileNetV2 model weights, label encoder, training history, metrics summary, test report, and confusion matrix |
| Notebook 07 | FP32 and quantized deployment models, quantization metrics, classification reports, size comparison, and speed comparison |

Generated data, models, and outputs are intentionally excluded from Git and can be recreated by running the notebooks in order.

## Evaluation Metrics

The implementation reports accuracy, macro F1, weighted F1, per-class precision, per-class recall, per-class F1, and confusion matrices. Quantization additionally reports model size, samples per second, milliseconds per sample, and percentage changes relative to the FP32 MobileNetV2 deployment artifact.

Macro F1 is the most important aggregate metric because it gives equal weight to each genre and is therefore more informative under the strong class imbalance introduced by the combined GTZAN and FMA Medium dataset.

## Limitations

- The final label space is limited to seven shared genres.
- The combined dataset is strongly imbalanced, especially toward `rock`.
- No data augmentation, class weighting for neural models, or segmentation-based training is implemented.
- Audio is converted to mono, so stereo information is discarded.
- Recordings are center-cropped to 28 seconds, which may remove useful information outside the central window.
- Some files are skipped during feature extraction because they cannot be decoded or processed.
- MobileNetV2 is pretrained on natural images, not audio spectrograms, so the transfer learning domain match is imperfect.
- Static INT8 quantization is not successful for predictive performance in the executed experiment.

## Reproducibility

Reproducibility is supported through:

- centralized paths and hyperparameters in `src/utils/config.py`,
- deterministic split creation with `random_state=42`,
- fixed shared genre mapping,
- saved metadata splits reused across all modeling notebooks,
- saved feature datasets reused by model notebooks,
- reusable source modules for preprocessing, feature extraction, model training, evaluation, and quantization.

To reproduce the full workflow, install the requirements, place the raw GTZAN and FMA files in the expected `data/raw/` structure, and execute notebooks 01 through 07 in order.
