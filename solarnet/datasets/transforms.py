import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional


def train_transforms(image_size: int = 256,
                     mean: tuple = (0.485, 0.456, 0.406),
                     std: tuple = (0.229, 0.224, 0.225)) -> alb.Compose:
    return alb.Compose([
        alb.RandomSizedCrop(min_max_height=(64, 200), height=image_size, width=image_size, p=0.5),
        alb.Flip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.OneOf([
            alb.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            alb.GridDistortion(p=0.5),
            alb.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
        ],
                  p=0.8),
        alb.CLAHE(p=0.8),
        alb.RandomBrightnessContrast(p=0.8),
        alb.RandomGamma(p=0.8),
        alb.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def test_transforms(mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> alb.Compose:
    return alb.Compose([alb.Normalize(mean=mean, std=std), ToTensorV2()])
