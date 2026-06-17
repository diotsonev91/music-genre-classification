# Project Proposal

## Project Title

Music Genre Classification using Machine Learning, Deep Learning, and Digital Signal Processing

## Motivation

Music genre classification is a representative task in music information retrieval because it requires models to learn relationships between acoustic structure and semantic genre labels. Audio recordings contain information across time, frequency, rhythm, timbre, harmony, and production style. These properties cannot be used directly by most conventional machine learning algorithms, which motivates the use of digital signal processing representations such as Mel-Frequency Cepstral Coefficients and Mel Spectrograms.

The project investigates whether classical machine learning models trained on compact MFCC summaries, convolutional neural networks trained on Mel Spectrograms, and transfer learning with MobileNetV2 can classify music recordings into shared genre categories. It also evaluates whether post-training quantization can reduce deployment cost while preserving predictive performance.

## Problem Statement

The problem is to build and evaluate a reproducible music genre classification pipeline using two public music datasets with different taxonomies and data characteristics. GTZAN provides short balanced genre excerpts, while FMA Medium provides a larger and more imbalanced real-world collection. The project must harmonize both datasets into a shared label space, generate consistent train, validation, and test splits, extract audio features, train multiple model families, compare their results, and assess deployment-oriented quantization.

## Objectives

- Explore GTZAN and FMA Medium audio and metadata characteristics.
- Harmonize the datasets through a shared genre mapping.
- Create reproducible stratified train, validation, and test splits.
- Standardize audio through resampling, mono conversion, fixed-duration center cropping, and optional normalization.
- Extract MFCC feature vectors for classical machine learning models.
- Extract Mel Spectrogram tensors for convolutional and transfer learning models.
- Train and evaluate SVM and Random Forest baselines.
- Train and evaluate a small CNN baseline.
- Train and evaluate an ImageNet-pretrained MobileNetV2 transfer learning model.
- Apply post-training quantization to the MobileNetV2 model and compare predictive and deployment metrics.
- Save reusable datasets, models, reports, and visual artifacts.

## Research Questions

1. Can compact MFCC summary features support reliable genre classification on the harmonized GTZAN and FMA Medium dataset?
2. Do Mel Spectrogram tensors provide sufficient structure for a CNN trained from scratch to learn genre-discriminative patterns?
3. Does transfer learning with MobileNetV2 improve over the CNN baseline on Mel Spectrogram inputs?
4. How does class imbalance affect macro F1, weighted F1, and per-class performance?
5. Does validation performance generalize consistently to the held-out test split?
6. Can post-training quantization reduce deployment cost without unacceptable loss of macro F1?

## Hypotheses

| ID | Hypothesis | Operational Criterion |
|---|---|---|
| H1 | Classical models trained on MFCC features can provide a useful baseline. | At least one classical model achieves validation macro F1 above 0.50. |
| H2 | Mel Spectrograms are suitable inputs for a CNN trained from scratch. | The CNN achieves validation macro F1 above 0.50. |
| H3 | The CNN can match or outperform the best classical baseline. | The CNN validation macro F1 is within 0.05 of, or higher than, the best classical validation macro F1. |
| H4 | Class imbalance affects model evaluation. | Macro F1 is lower than weighted F1 and per-class reports show uneven performance. |
| H5 | The selected model generalizes from validation to test data. | Test macro F1 is no more than 0.10 below validation macro F1. |
| H6 | MobileNetV2 transfer learning improves over the CNN baseline. | MobileNetV2 validation macro F1 is higher than the CNN baseline, or comparable within 0.05 using fewer epochs. |
| H7 | Quantization can improve deployment efficiency while preserving performance. | At least one quantized MobileNetV2 variant provides a size or latency benefit while keeping macro F1 within 0.05 of the FP32 model. |

## Datasets

### GTZAN Genre Collection

GTZAN is used as the primary benchmark dataset for baseline experiments and model evaluation. The implementation expects the dataset under `data/raw/gtzan/genres/`, with one folder per genre and `.au` audio files inside each genre folder. After shared-genre filtering, the project uses the GTZAN labels `blues`, `classical`, `country`, `hiphop`, `jazz`, `pop`, and `rock`.

Download source: GTZAN Genre Collection on Kaggle.

### FMA Medium Dataset

FMA Medium is used to increase dataset size and evaluate model behavior under a more realistic, imbalanced distribution. The implementation expects the audio archive under `data/raw/fma_medium/` and metadata under `data/raw/fma_metadata/`. The metadata file `tracks.csv` is used to select the `medium` subset and map FMA top-level genres into the shared label space.

Required downloads:

- `fma_medium.zip`
- `fma_metadata.zip`

Source: FMA official repository.

## Expected Workflow

1. Explore dataset organization, class distributions, audio duration, and signal representations in Notebook 01.
2. Filter and harmonize GTZAN and FMA Medium into shared genres in Notebook 02.
3. Create stratified train, validation, and test metadata splits in Notebook 02.
4. Extract MFCC CSV datasets and Mel Spectrogram tensor datasets in Notebook 03.
5. Train SVM and Random Forest baselines on MFCC features in Notebook 04.
6. Train a CNN baseline on Mel Spectrogram tensors in Notebook 05.
7. Train a frozen-feature MobileNetV2 transfer learning model in Notebook 06.
8. Quantize the saved MobileNetV2 model and compare deployment metrics in Notebook 07.

## Evaluation Metrics

The project reports:

- Accuracy
- Macro F1
- Weighted F1
- Precision, recall, and F1 score by class
- Confusion matrices
- Model size for deployment artifacts
- CPU inference latency and throughput for quantized variants

Macro F1 is the primary model selection and hypothesis metric because the combined dataset is strongly imbalanced.

## Expected Deliverables

- Dataset exploration and preprocessing notebooks.
- Reproducible metadata splits.
- MFCC feature CSV files.
- Mel Spectrogram `.npy` feature and label files.
- Trained classical, CNN, and MobileNetV2 model artifacts.
- Evaluation summaries, classification reports, and confusion matrices.
- Quantized deployment model artifacts and deployment comparison reports.
- Documentation describing the proposal, methodology, repository usage, results, limitations, and reproducibility.
