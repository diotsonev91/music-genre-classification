from pathlib import Path
import sys
import types

import numpy as np
import pytest

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
    def _fake_frames(
        y,
        hop_length
    ):
        return max(
            1,
            1 + len(y) // hop_length
        )

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
    _librosa.amplitude_to_db = lambda values, ref=np.max: np.asarray(
        values,
        dtype=np.float32
    )
    _librosa.power_to_db = lambda values, ref=np.max: np.asarray(
        values,
        dtype=np.float32
    )
    _librosa.feature = types.SimpleNamespace(
        melspectrogram=lambda y, sr, n_mels=128, n_fft=2048, hop_length=512: np.ones(
            (
                n_mels,
                _fake_frames(
                    y,
                    hop_length
                )
            ),
            dtype=np.float32
        ),
        mfcc=lambda y, sr, n_mfcc=20, n_mels=128, n_fft=2048, hop_length=512: np.ones(
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

from src.data import preprocessing


def test_is_long_enough_uses_sample_count_duration():
    sr = 10

    assert preprocessing.is_long_enough(
        np.zeros(20),
        sr,
        min_duration=2.0
    )

    assert not preprocessing.is_long_enough(
        np.zeros(19),
        sr,
        min_duration=2.0
    )


def test_crop_audio_center_and_start_modes():
    y = np.arange(
        10
    )

    centered = preprocessing.crop_audio(
        y,
        sr=2,
        duration=2.0,
        mode="center"
    )

    started = preprocessing.crop_audio(
        y,
        sr=2,
        duration=2.0,
        mode="start"
    )

    np.testing.assert_array_equal(
        centered,
        np.array([3, 4, 5, 6])
    )

    np.testing.assert_array_equal(
        started,
        np.array([0, 1, 2, 3])
    )


def test_crop_audio_rejects_short_audio_and_unknown_mode():
    y = np.arange(
        5
    )

    with pytest.raises(
        ValueError,
        match="shorter"
    ):
        preprocessing.crop_audio(
            y,
            sr=10,
            duration=1.0
        )

    with pytest.raises(
        ValueError,
        match="Unsupported"
    ):
        preprocessing.crop_audio(
            y,
            sr=10,
            duration=0.2,
            mode="end"
        )


def test_normalize_audio_peak_scales_by_max_abs_value():
    y = np.array(
        [-2.0, 0.0, 1.0]
    )

    normalized = preprocessing.normalize_audio(
        y,
        method="peak"
    )

    np.testing.assert_allclose(
        normalized,
        np.array([-1.0, 0.0, 0.5])
    )

    assert normalized.dtype == y.dtype


def test_normalize_audio_none_and_silent_signal_are_unchanged():
    y = np.array(
        [0.0, 0.0, 0.0]
    )

    assert preprocessing.normalize_audio(
        y,
        method=None
    ) is y

    np.testing.assert_array_equal(
        preprocessing.normalize_audio(
            y,
            method="peak"
        ),
        y
    )


def test_normalize_audio_rejects_unknown_method():
    with pytest.raises(
        ValueError,
        match="Unsupported"
    ):
        preprocessing.normalize_audio(
            np.array([1.0]),
            method="rms"
        )


def test_preprocess_audio_returns_none_for_short_audio(monkeypatch):
    def fake_load_audio(filepath, sample_rate, mono):
        return np.zeros(4), sample_rate

    monkeypatch.setattr(
        preprocessing,
        "load_audio",
        fake_load_audio
    )

    y, sr = preprocessing.preprocess_audio(
        "unused.wav",
        sample_rate=10,
        duration=1.0
    )

    assert y is None
    assert sr is None


def test_preprocess_audio_crops_and_normalizes_loaded_audio(monkeypatch):
    def fake_load_audio(filepath, sample_rate, mono):
        assert filepath == "synthetic.wav"
        assert sample_rate == 4
        assert mono

        return np.array(
            [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
        ), sample_rate

    monkeypatch.setattr(
        preprocessing,
        "load_audio",
        fake_load_audio
    )

    y, sr = preprocessing.preprocess_audio(
        "synthetic.wav",
        sample_rate=4,
        mono=True,
        duration=1.0,
        crop_mode="center",
        normalization="peak"
    )

    assert sr == 4
    assert isinstance(
        y,
        np.ndarray
    )
    assert y.shape == (
        4,
    )

    np.testing.assert_allclose(
        y,
        np.array([-1.0, 0.0, 1.0, 2.0]) / 2.0
    )
