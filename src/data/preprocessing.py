import numpy as np
import librosa


def load_audio(
    filepath,
    sample_rate=22050,
    mono=True
):
    """
    Load audio file.
    """

    y, sr = librosa.load(
        filepath,
        sr=sample_rate,
        mono=mono
    )

    return y, sr


def is_long_enough(
    y,
    sr,
    min_duration=28.0
):
    """
    Check whether audio duration
    is sufficient.
    """

    duration = len(y) / sr

    return duration >= min_duration


def crop_audio(
    y,
    sr,
    duration=28.0,
    mode="center"
):
    """
    Crop audio to fixed duration.

    No padding is applied.
    """

    target = int(
        sr * duration
    )

    if len(y) < target:

        raise ValueError(
            "Audio shorter than target duration."
        )

    if mode == "center":

        start = (
            len(y)
            - target
        ) // 2

    elif mode == "start":

        start = 0

    else:

        raise ValueError(
            "Unsupported crop mode."
        )

    return y[
        start:
        start + target
    ]


def normalize_audio(
    y,
    method=None,
    eps=1e-9
):
    """
    Optional normalization.
    """

    if method is None:

        return y


    if method == "peak":

        peak = np.max(
            np.abs(y)
        )

        if peak < eps:

            return y

        return y / peak


    raise ValueError(
        "Unsupported normalization."
    )


def preprocess_audio(
    filepath,
    sample_rate=22050,
    mono=True,
    duration=28.0,
    crop_mode="center",
    normalization=None
):
    """
    Complete preprocessing pipeline.
    """

    y, sr = load_audio(
        filepath,
        sample_rate,
        mono
    )

    valid = is_long_enough(
        y,
        sr,
        duration
    )

    if not valid:

        return None, None


    y = crop_audio(
        y,
        sr,
        duration,
        crop_mode
    )

    y = normalize_audio(
        y,
        normalization
    )

    return y, sr