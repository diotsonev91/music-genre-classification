import torch
from torch import nn


class MusicGenreCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes
    ):
        super().__init__()

        channels = input_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(
                channels,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(
                16
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                2
            ),
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(
                32
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                2
            ),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(
                64
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64,
                64
            ),
            nn.ReLU(),
            nn.Dropout(
                0.3
            ),
            nn.Linear(
                64,
                num_classes
            )
        )

        self.learning_rate = None

    def forward(
        self,
        x
    ):
        x = self.features(
            x
        )

        return self.classifier(
            x
        )


def create_cnn_model(
    input_shape,
    num_classes,
    learning_rate=0.001
):
    """
    Create a small CNN baseline for Mel Spectrogram inputs.

    The model returns raw logits. Training should use
    torch.nn.CrossEntropyLoss, which applies softmax internally.
    """

    model = MusicGenreCNN(
        input_shape=input_shape,
        num_classes=num_classes
    )

    model.learning_rate = learning_rate

    return model
