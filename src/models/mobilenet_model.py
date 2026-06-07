import torch
from torch import nn
from torch.nn import functional as F


class MelMobileNetV2(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        train_base=False
    ):
        super().__init__()

        try:
            from torchvision.models import (
                MobileNet_V2_Weights,
                mobilenet_v2
            )

        except ImportError as error:
            raise ImportError(
                "MobileNetV2 requires torchvision. "
                "Install the project requirements before running Notebook 06."
            ) from error

        if input_shape[0] not in (
            1,
            3
        ):
            raise ValueError(
                "MobileNetV2 expects one-channel or three-channel input."
            )

        self.input_channels = input_shape[0]
        self.train_base = train_base

        self.model = mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT
        )

        for parameter in self.model.features.parameters():
            parameter.requires_grad = train_base

        input_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(
                0.2
            ),
            nn.Linear(
                input_features,
                num_classes
            )
        )

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(
                1,
                3,
                1,
                1
            )
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(
                1,
                3,
                1,
                1
            )
        )

        self.learning_rate = None

    def train(
        self,
        mode=True
    ):
        super().train(
            mode
        )

        if not self.train_base:
            self.model.features.eval()

        return self

    def forward(
        self,
        x
    ):
        if x.shape[1] == 1:
            x = x.repeat(
                1,
                3,
                1,
                1
            )

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        x = (
            x
            - self.image_mean
        ) / self.image_std

        return self.model(
            x
        )


def create_mobilenet_model(
    input_shape,
    num_classes,
    learning_rate,
    train_base=False
):
    """
    Create an ImageNet-pretrained MobileNetV2 classifier.

    Single-channel Mel Spectrogram batches are converted to normalized
    three-channel 224 x 224 inputs inside the model.
    """

    model = MelMobileNetV2(
        input_shape=input_shape,
        num_classes=num_classes,
        train_base=train_base
    )

    model.learning_rate = learning_rate

    return model
