import albumentations as alb
from pathlib import Path
from solarnet.datasets import USGSClassifierDataset, USGSSegmentationDataset


def test_classifier_dataset(data_folder: Path):
    print(data_folder)

    transform = alb.Compose([
        alb.RandomCrop(width=256, height=256),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
    ])
    dataset = USGSClassifierDataset(data_folder=data_folder, transform=transform)
    x, y = dataset.__getitem__(0)
    assert x.shape == (256, 256, 3)
    assert y >= 0 and y <= 1


def test_usgs_segmentation_dataset(data_folder: Path):
    transform = alb.Compose([
        alb.RandomCrop(width=256, height=256),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
    ])
    dataset = USGSSegmentationDataset(data_folder=data_folder, transform=transform)
    x, y = dataset.__getitem__(0)
    print(x.min(), x.max())
    assert x.shape == (256, 256, 3)
    assert y.shape == (256, 256)


def test_dydas_segmentation_dataset():
    pass
