import numpy as np
import librosa
import pandas as pd
from pathlib import Path


def compute_stft(y, n_fft=2048, hop_length=512):
    """
    Compute Short-Time Fourier Transform (STFT).

    STFT splits the signal into short overlapping windows
    and applies Fourier Transform to each window.

    This produces a time-frequency representation showing
    how frequency content changes over time.
    """

    stft = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length
    )

    stft_db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    return stft_db


def compute_mel_spectrogram(
    y,
    sr,
    n_mels=128,
    n_fft=2048,
    hop_length=512
):
    """
    Compute Mel Spectrogram.

    A Mel Spectrogram maps frequency energy onto the Mel scale,
    which approximates human perception of pitch.

    This representation is useful for CNN models because it
    behaves similarly to an image with time and frequency axes.
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


def standardize_time_frames(
    spectrogram,
    target_frames
):
    current_frames = spectrogram.shape[1]

    if current_frames == target_frames:

        return spectrogram

    if current_frames > target_frames:

        start = (
            current_frames
            - target_frames
        ) // 2

        return spectrogram[
            :,
            start:
            start + target_frames
        ]

    padding = target_frames - current_frames

    return np.pad(
        spectrogram,
        (
            (0, 0),
            (0, padding)
        ),
        mode="constant",
        constant_values=np.min(
            spectrogram
        )
    )


def normalize_mel_spectrogram(
    mel,
    min_db=-80.0,
    max_db=0.0
):
    mel = np.clip(
        mel,
        min_db,
        max_db
    )

    return (
        mel
        - min_db
    ) / (
        max_db
        - min_db
    )


def compute_mfcc(
    y,
    sr,
    n_mfcc=20,
    n_mels=128,
    n_fft=2048,
    hop_length=512
):
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCC).

    MFCCs summarize the spectral envelope of an audio signal.

    They are commonly used as compact timbre-related features
    for classical Machine Learning models.
    """

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    return mfcc



def aggregate_mfcc(
    mfcc
):
    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    return np.concatenate([
        mfcc_mean,
        mfcc_std
    ])


def extract_mfcc_dataset(
    metadata,
    preprocess_fn,
    sample_rate,
    mono,
    duration,
    crop_mode,
    normalization
):
    rows = []

    for _, row in metadata.iterrows():

        try:
            y, sr = preprocess_fn(
                row["filepath"],
                sample_rate,
                mono,
                duration,
                crop_mode,
                normalization
            )

            mfcc = compute_mfcc(
                y,
                sr
            )

            vector = aggregate_mfcc(
                mfcc
            )

            sample = {
                "label": row["label"]
            }

            for i, value in enumerate(
                vector
            ):
                sample[f"mfcc_{i}"] = value

            rows.append(sample)

        except Exception:
            continue

    return pd.DataFrame(rows)


def extract_mel_dataset(
    metadata,
    preprocess_fn,
    sample_rate,
    mono,
    duration,
    crop_mode,
    normalization,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    target_frames=None,
    add_channel=True,
    normalize=True
):
    features = []
    labels = []
    skipped = []

    for _, row in metadata.iterrows():

        try:
            y, sr = preprocess_fn(
                row["filepath"],
                sample_rate,
                mono,
                duration,
                crop_mode,
                normalization
            )

            if y is None:

                skipped.append(
                    row["filepath"]
                )

                continue

            mel = compute_mel_spectrogram(
                y,
                sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )

            if target_frames is not None:

                mel = standardize_time_frames(
                    mel,
                    target_frames
                )

            if normalize:

                mel = normalize_mel_spectrogram(
                    mel
                )

            if add_channel:

                mel = np.expand_dims(
                    mel,
                    axis=0
                )

            features.append(
                mel.astype(
                    np.float32
                )
            )

            labels.append(
                row["label"]
            )

        except Exception:
            skipped.append(
                row["filepath"]
            )

            continue

    if not features:

        if target_frames is None:

            target_frames = 0

        if add_channel:

            empty_shape = (
                0,
                1,
                n_mels,
                target_frames
            )

        else:

            empty_shape = (
                0,
                n_mels,
                target_frames
            )

        return (
            np.empty(
                empty_shape,
                dtype=np.float32
            ),
            np.array(
                labels
            ),
            skipped
        )

    return (
        np.stack(
            features
        ),
        np.array(
            labels
        ),
        skipped
    )


def save_mel_dataset(
    features,
    labels,
    features_path,
    labels_path
):
    features_path = Path(
        features_path
    )

    labels_path = Path(
        labels_path
    )

    features_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    labels_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        features_path,
        features
    )

    np.save(
        labels_path,
        labels
    )


def load_mel_dataset(
    features_path,
    labels_path
):
    features_path = Path(
        features_path
    )

    labels_path = Path(
        labels_path
    )

    return (
        np.load(
            features_path
        ),
        np.load(
            labels_path,
            allow_pickle=True
        )
    )
