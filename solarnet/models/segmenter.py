import torch
from torch import nn

from typing import List

from .base import ResnetBase


class Segmenter(ResnetBase):
    """A ResNet34 U-Net model, as described in
    https://github.com/fastai/fastai/blob/master/courses/dl2/carvana-unet-lrg.ipynb

    Attributes:
        imagenet_base: boolean, default: False
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, imagenet_base: bool = False) -> None:
        super().__init__(imagenet_base=imagenet_base)

        self.target_modules = [str(x) for x in [2, 4, 5, 6]]
        self.hooks = self.add_hooks()

        self.relu = nn.ReLU()
        self.upsamples = nn.ModuleList([
            UpBlock(2048, 1024, 512),
            UpBlock(512, 512, 256),
            UpBlock(256, 256, 64),
            UpBlock(64, 64, 32),
            UpBlock(32, 3, 16),
        ])
        self.conv_transpose = nn.ConvTranspose2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def add_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks = []
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                hooks.append(child.register_forward_hook(self.save_output))
        return hooks

    def retrieve_hooked_outputs(self) -> List[torch.Tensor]:
        # to be called in the forward pass, this method returns the tensors
        # which were saved by the forward hooks
        outputs = []
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                outputs.append(child.output)
        return outputs

    def cleanup(self) -> None:
        # removes the hooks, and the tensors which were added
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                # allows the method to be safely called even if
                # the hooks aren't there
                try:
                    del child.output
                except AttributeError:
                    continue
        for hook in self.hooks:
            hook.remove()

    @staticmethod
    def save_output(module, input, output):
        # the hook to add to the target modules
        module.output = output

    def load_base(self, state_dict: dict) -> None:
        # This allows a model trained on the classifier to be loaded
        # into the model used for segmentation, even though their state_dicts
        # differ
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        org_input = x
        x = self.relu(self.pretrained(x))
        # we reverse the outputs so that the smallest output
        # is the first one we get, and the largest the last
        interim = self.retrieve_hooked_outputs()[::-1]

        for upsampler, interim_output in zip(self.upsamples[:-1], interim):
            x = upsampler(x, interim_output)
        x = self.upsamples[-1](x, org_input)
        return self.sigmoid(self.conv_transpose(x))


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, across_channels: int, out_channels: int) -> None:
        super().__init__()
        up_out = across_out = out_channels // 2
        self.conv_across = nn.Conv2d(across_channels, across_out, 1)
        # alternative: ConvTranspose2d(in_channels, up_out, 2, stride=2)
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                      nn.Conv2d(in_channels, up_out, kernel_size=1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x_up, x_across):
        upsampled = self.upsample(x_up)
        skipped = self.conv_across(x_across)
        joint = torch.cat((upsampled, skipped), dim=1)
        return self.batchnorm(self.relu(joint))
