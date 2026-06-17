from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


try:
    import librosa as _librosa
except ModuleNotFoundError:
    _librosa = types.ModuleType(
        "librosa"
    )
    sys.modules[
        "librosa"
    ] = _librosa

def _fake_frames(
    y,
    hop_length
):
    return max(
        1,
        1 + len(y) // hop_length
    )


if not hasattr(
    _librosa,
    "load"
):
    _librosa.load = lambda filepath, sr=22050, mono=True: (
        np.zeros(
            sr,
            dtype=np.float32
        ),
        sr
    )

if not hasattr(
    _librosa,
    "stft"
):
    _librosa.stft = lambda y, n_fft=2048, hop_length=512: np.ones(
        (
            n_fft // 2 + 1,
            _fake_frames(
                y,
                hop_length
            )
        ),
        dtype=np.complex64
    )

if not hasattr(
    _librosa,
    "amplitude_to_db"
):
    _librosa.amplitude_to_db = lambda values, ref=np.max: np.asarray(
        values,
        dtype=np.float32
    )

if not hasattr(
    _librosa,
    "power_to_db"
):
    _librosa.power_to_db = lambda values, ref=np.max: np.asarray(
        values,
        dtype=np.float32
    )

if not hasattr(
    _librosa,
    "feature"
):
    _librosa.feature = types.SimpleNamespace()

if not hasattr(
    _librosa.feature,
    "melspectrogram"
):
    _librosa.feature.melspectrogram = (
        lambda y, sr, n_mels=128, n_fft=2048, hop_length=512: np.ones(
            (
                n_mels,
                _fake_frames(
                    y,
                    hop_length
                )
            ),
            dtype=np.float32
        )
    )

if not hasattr(
    _librosa.feature,
    "mfcc"
):
    _librosa.feature.mfcc = (
        lambda y, sr, n_mfcc=20, n_mels=128, n_fft=2048, hop_length=512: np.ones(
            (
                n_mfcc,
                _fake_frames(
                    y,
                    hop_length
                )
            ),
            dtype=np.float32
        )
    )

from src.data import feature_extraction


def test_standardize_time_frames_returns_same_shape_when_already_matching():
    spectrogram = np.arange(
        6
    ).reshape(
        2,
        3
    )

    result = feature_extraction.standardize_time_frames(
        spectrogram,
        target_frames=3
    )

    assert result is spectrogram


def test_standardize_time_frames_center_crops_long_spectrogram():
    spectrogram = np.arange(
        12
    ).reshape(
        2,
        6
    )

    result = feature_extraction.standardize_time_frames(
        spectrogram,
        target_frames=4
    )

    assert result.shape == (
        2,
        4
    )

    np.testing.assert_array_equal(
        result,
        spectrogram[:, 1:5]
    )


def test_standardize_time_frames_pads_short_spectrogram_with_min_value():
    spectrogram = np.array([
        [2.0, 3.0],
        [4.0, 5.0],
    ])

    result = feature_extraction.standardize_time_frames(
        spectrogram,
        target_frames=5
    )

    assert result.shape == (
        2,
        5
    )

    np.testing.assert_array_equal(
        result[:, :2],
        spectrogram
    )

    np.testing.assert_array_equal(
        result[:, 2:],
        np.full(
            (2, 3),
            2.0
        )
    )


def test_normalize_mel_spectrogram_clips_and_scales_to_unit_range():
    mel = np.array([
        [-100.0, -80.0, -40.0, 0.0, 10.0]
    ])

    normalized = feature_extraction.normalize_mel_spectrogram(
        mel,
        min_db=-80.0,
        max_db=0.0
    )

    np.testing.assert_allclose(
        normalized,
        np.array([
            [0.0, 0.0, 0.5, 1.0, 1.0]
        ])
    )


def test_aggregate_mfcc_concatenates_mean_and_standard_deviation():
    mfcc = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 2.0],
    ])

    result = feature_extraction.aggregate_mfcc(
        mfcc
    )

    assert result.shape == (
        4,
    )

    np.testing.assert_allclose(
        result[:2],
        np.array([2.0, 2.0])
    )

    np.testing.assert_allclose(
        result[2:],
        np.array([
            np.std([1.0, 2.0, 3.0]),
            0.0,
        ])
    )


def test_dsp_feature_functions_return_expected_shapes_and_finite_values():
    sr = 8000
    time = np.arange(
        256
    ) / sr

    y = np.sin(
        2 * np.pi * 440 * time
    ).astype(
        np.float32
    )

    stft = feature_extraction.compute_stft(
        y,
        n_fft=32,
        hop_length=16
    )

    mel = feature_extraction.compute_mel_spectrogram(
        y,
        sr,
        n_mels=8,
        n_fft=32,
        hop_length=16
    )

    mfcc = feature_extraction.compute_mfcc(
        y,
        sr,
        n_mfcc=4,
        n_mels=8,
        n_fft=32,
        hop_length=16
    )

    assert stft.shape[0] == 17
    assert mel.shape[0] == 8
    assert mfcc.shape[0] == 4

    assert np.isfinite(
        stft
    ).all()
    assert np.isfinite(
        mel
    ).all()
    assert np.isfinite(
        mfcc
    ).all()


def test_extract_mel_dataset_uses_preprocess_output_and_tracks_skips(monkeypatch):
    metadata = pd.DataFrame([
        {
            "filepath": "valid.wav",
            "label": "rock",
        },
        {
            "filepath": "short.wav",
            "label": "jazz",
        },
        {
            "filepath": "broken.wav",
            "label": "pop",
        },
    ])

    def fake_preprocess(
        filepath,
        sample_rate,
        mono,
        duration,
        crop_mode,
        normalization
    ):
        if filepath == "valid.wav":
            return np.ones(
                16,
                dtype=np.float32
            ), sample_rate

        if filepath == "short.wav":
            return None, None

        raise RuntimeError(
            "decode failed"
        )

    def fake_compute_mel(
        y,
        sr,
        n_mels,
        n_fft,
        hop_length
    ):
        return np.arange(
            n_mels * 3,
            dtype=np.float32
        ).reshape(
            n_mels,
            3
        )

    monkeypatch.setattr(
        feature_extraction,
        "compute_mel_spectrogram",
        fake_compute_mel
    )

    features, labels, skipped = feature_extraction.extract_mel_dataset(
        metadata,
        fake_preprocess,
        sample_rate=8000,
        mono=True,
        duration=1.0,
        crop_mode="center",
        normalization=None,
        n_mels=4,
        n_fft=32,
        hop_length=16,
        target_frames=5,
        add_channel=True,
        normalize=False
    )

    assert features.shape == (
        1,
        1,
        4,
        5
    )
    assert features.dtype == np.float32

    np.testing.assert_array_equal(
        labels,
        np.array(["rock"])
    )

    assert skipped == [
        "short.wav",
        "broken.wav",
    ]


def test_extract_mel_dataset_returns_empty_array_when_all_rows_skip():
    metadata = pd.DataFrame([
        {
            "filepath": "short.wav",
            "label": "rock",
        }
    ])

    def fake_preprocess(
        filepath,
        sample_rate,
        mono,
        duration,
        crop_mode,
        normalization
    ):
        return None, None

    features, labels, skipped = feature_extraction.extract_mel_dataset(
        metadata,
        fake_preprocess,
        sample_rate=8000,
        mono=True,
        duration=1.0,
        crop_mode="center",
        normalization=None,
        n_mels=4,
        target_frames=6,
        add_channel=False
    )

    assert features.shape == (
        0,
        4,
        6
    )
    assert features.dtype == np.float32
    assert labels.shape == (
        0,
    )
    assert skipped == [
        "short.wav"
    ]


def test_save_and_load_mel_dataset_round_trip(tmp_path):
    features = np.ones(
        (2, 1, 4, 5),
        dtype=np.float32
    )
    labels = np.array([
        "rock",
        "jazz",
    ])

    features_path = tmp_path / "features.npy"
    labels_path = tmp_path / "labels.npy"

    feature_extraction.save_mel_dataset(
        features,
        labels,
        features_path,
        labels_path
    )

    loaded_features, loaded_labels = feature_extraction.load_mel_dataset(
        features_path,
        labels_path
    )

    np.testing.assert_array_equal(
        loaded_features,
        features
    )

    np.testing.assert_array_equal(
        loaded_labels,
        labels
    )
