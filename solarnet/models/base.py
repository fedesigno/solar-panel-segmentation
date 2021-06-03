import torch
from torch import nn
from torchvision.models import resnet34, resnet50, resnet101

backbones = {"resnet34": resnet34, "resnet50": resnet50, "resnet101": resnet101}


class ResnetBase(nn.Module):
    """ResNet pretrained on Imagenet. This serves as the
    base for the classifier, and subsequently the segmentation model

    Attributes:
        imagenet_base: boolean, default: True
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, version: str = "resnet50", imagenet_base: bool = True) -> None:
        super().__init__()
        try:
            backbone_fn = backbones[version]
        except KeyError as e:
            ValueError(f"Error while looking for the backbone '{version}': {str(e)}")

        resnet = backbone_fn(pretrained=imagenet_base).float()
        self.pretrained = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # Since this is just a base, forward() shouldn't directly
        # be called on it.
        raise NotImplementedError
