# Project Proposal

## Title

Music Genre Classification using Machine Learning, Deep Learning, and DSP

---

## Datasets

### GTZAN Dataset
Source:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Description:
- 10 music genres
- 100 audio files per genre
- ~30 second excerpts
- Used as part of the combined training, validation, and test dataset after shared genre filtering

---

### FMA Medium Dataset
Source:
https://github.com/mdeff/fma

Audio:
https://os.unil.cloud.switch.ch/fma/fma_medium.zip

Metadata:
https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

Description:
- 25,000 tracks
- 16 music genres
- Variable duration recordings
- Used together with GTZAN after selecting shared genres
- Contributes larger scale and more realistic class imbalance

---

## Objectives

- Compare MFCC and Mel Spectrogram representations
- Compare Classical Machine Learning and Deep Learning approaches
- Evaluate post-training quantization
- Evaluate model generalization on the held-out combined test set

---

## Research Hypotheses

The following hypotheses are evaluated using the existing train, validation, and test experiments:

- **H1 — Classical baseline performance:** At least one classical model trained on MFCC features will achieve a validation macro F1 score above 0.50.
- **H2 — Mel Spectrogram suitability:** A CNN trained on Mel Spectrogram tensors will achieve a validation macro F1 score above 0.50, showing that this representation contains useful genre information.
- **H3 — CNN comparison:** The CNN will match the best classical model within 0.05 macro F1, or outperform it, on the validation split.
- **H4 — Class imbalance effect:** Macro F1 will be lower than weighted F1, and per-class results will vary substantially, indicating stronger performance on frequent genres than on minority genres.
- **H5 — Generalization:** For the selected model, test macro F1 will be no more than 0.10 below validation macro F1.
- **H6 — Transfer Learning Advantage:** MobileNetV2 transfer learning will achieve a higher validation macro F1 score than the baseline CNN, or comparable performance within 0.05 macro F1 using fewer training epochs.

These operational criteria allow each hypothesis to be accepted or rejected directly from the existing metric summaries and classification reports.

---

## Planned Models

Classical:
- SVM
- Random Forest

Deep Learning:
- CNN
- MobileNetV2

---

## Evaluation Strategy

The project uses a combined dataset strategy.

GTZAN and FMA Medium are first harmonized using shared genre labels.

The combined dataset is then split into:

- Training set
- Validation set
- Test set

The validation set is used for model selection.

The test set is used only once for final evaluation of the selected model.

Metrics:
- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1
- Confusion Matrix
