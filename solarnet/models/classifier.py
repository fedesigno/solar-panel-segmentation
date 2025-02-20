import torch
from torch import nn

from .base import ResnetBase


class Classifier(ResnetBase):
    """A ResNet34 Model

    Attributes:
        imagenet_base: boolean, default: True
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, imagenet_base: bool = True, backbone: str = "resnet50") -> None:
        super().__init__(imagenet_base=imagenet_base, version=backbone)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.classifier = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.pretrained(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))
