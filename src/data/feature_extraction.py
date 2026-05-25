import numpy as np
import librosa
import pandas as pd


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