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
- Used for training and validation

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
- Used only for final external evaluation

---

## Objectives

- Compare MFCC and Mel Spectrogram representations
- Compare Classical Machine Learning and Deep Learning approaches
- Evaluate post-training quantization
- Evaluate model generalization on an external dataset

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

Training:
- GTZAN

Validation:
- GTZAN validation split

External Testing:
- FMA Medium (shared genres only)

Metrics:
- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1