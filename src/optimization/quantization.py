import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch.ao.quantization import (
    get_default_qconfig_mapping,
    quantize_dynamic
)
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    prepare_fx
)


def select_quantization_backend():
    supported_engines = torch.backends.quantized.supported_engines

    for candidate in [
        "x86",
        "fbgemm",
        "onednn",
        "qnnpack",
    ]:
        if candidate in supported_engines:
            return candidate

    raise RuntimeError(
        f"No supported quantization backend found: {supported_engines}"
    )


def prepare_mobilenet_inner_inputs(
    model,
    X_batch
):
    X_batch = torch.as_tensor(
        X_batch,
        dtype=torch.float32
    )

    if X_batch.ndim == 3:
        X_batch = X_batch.unsqueeze(
            0
        )

    if X_batch.shape[1] == 1:
        X_batch = X_batch.repeat(
            1,
            3,
            1,
            1
        )

    X_batch = F.interpolate(
        X_batch,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    image_mean = model.image_mean.detach().cpu()
    image_std = model.image_std.detach().cpu()

    return (
        X_batch
        - image_mean
    ) / image_std


def iter_calibration_batches(
    model,
    X,
    sample_count,
    batch_size
):
    sample_count = min(
        sample_count,
        len(X)
    )

    for start in range(
        0,
        sample_count,
        batch_size
    ):
        end = min(
            start + batch_size,
            sample_count
        )

        yield prepare_mobilenet_inner_inputs(
            model,
            X[start:end]
        )


def clone_fp32_model(
    model
):
    return copy.deepcopy(
        model
    ).cpu().eval()


def save_deployment_model(
    model,
    path,
    example_input
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model = model.cpu().eval()

    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                example_input.cpu(),
                strict=False
            )

            traced_model = torch.jit.freeze(
                traced_model
            )

        traced_model.save(
            str(path)
        )

        return "torchscript"

    except Exception as error:
        torch.save(
            model,
            path
        )

        print(
            "TorchScript export failed. Saved a PyTorch module instead."
        )
        print(
            type(error).__name__,
            error
        )

        return "pytorch_module"


def build_static_int8_model(
    base_model,
    X_calibration,
    quantization_backend,
    calibration_sample_count,
    batch_size
):
    mobilenet_core = copy.deepcopy(
        base_model.model
    ).cpu().eval()

    example_inner_input = prepare_mobilenet_inner_inputs(
        base_model,
        X_calibration[:1]
    )

    qconfig_mapping = get_default_qconfig_mapping(
        quantization_backend
    )

    prepared_core = prepare_fx(
        mobilenet_core,
        qconfig_mapping,
        example_inputs=(example_inner_input,)
    )

    with torch.no_grad():
        for calibration_batch in iter_calibration_batches(
            base_model,
            X_calibration,
            calibration_sample_count,
            batch_size
        ):
            prepared_core(
                calibration_batch
            )

    quantized_core = convert_fx(
        prepared_core
    )

    quantized_model = clone_fp32_model(
        base_model
    )

    quantized_model.model = quantized_core

    # FX GraphModule may not expose the original .features attribute.
    quantized_model.train_base = True

    return quantized_model


def build_dynamic_linear_model(
    base_model
):
    dynamic_model = clone_fp32_model(
        base_model
    )

    return quantize_dynamic(
        dynamic_model,
        {nn.Linear},
        dtype=torch.qint8
    )


def build_classifier_dynamic_model(
    base_model
):
    classifier_model = clone_fp32_model(
        base_model
    )

    classifier_model.model.classifier = quantize_dynamic(
        copy.deepcopy(
            classifier_model.model.classifier
        ).cpu().eval(),
        {nn.Linear},
        dtype=torch.qint8
    )

    return classifier_model


def size_in_mb(
    path
):
    return path.stat().st_size / (
        1024 ** 2
    )


def measure_inference_speed(
    model,
    X,
    batch_size,
    repeats=5,
    warmup=2
):
    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32
    )

    data_loader = DataLoader(
        TensorDataset(
            X_tensor
        ),
        batch_size=batch_size,
        shuffle=False
    )

    model = model.cpu().eval()

    with torch.no_grad():
        for _ in range(
            warmup
        ):
            for (X_batch,) in data_loader:
                model(
                    X_batch
                )

    elapsed_times = []
    sample_count = len(
        X_tensor
    )

    with torch.no_grad():
        for _ in range(
            repeats
        ):
            start_time = time.perf_counter()

            for (X_batch,) in data_loader:
                model(
                    X_batch
                )

            elapsed_times.append(
                time.perf_counter()
                - start_time
            )

    mean_seconds = float(
        np.mean(
            elapsed_times
        )
    )

    std_seconds = float(
        np.std(
            elapsed_times
        )
    )

    return {
        "samples": sample_count,
        "repeats": repeats,
        "mean_seconds": mean_seconds,
        "std_seconds": std_seconds,
        "samples_per_second": sample_count / mean_seconds,
        "milliseconds_per_sample": mean_seconds / sample_count * 1000,
    }
