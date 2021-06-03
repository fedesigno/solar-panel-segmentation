import numpy as np
import tifffile as tif
import torch
from pathlib import Path
import random

from torch.utils.data import Dataset
from typing import Callable, Optional, List, Tuple


class USGSSegmentationDataset:

    def __init__(self, data_folder: Path, transform: Callable = None, mask: Optional[List[bool]] = None) -> None:
        self.transform = transform

        # We will only segment the images which we know have solar panels in them; the
        # other images should be filtered out by the classifier
        solar_folder = data_folder / 'solar'
        self.org_solar_files = list((solar_folder / 'org').glob("*.tif"))
        self.mask_solar_files = [solar_folder / 'mask' / f.name for f in self.org_solar_files]
        assert len(self.org_solar_files) > 0, "No images found!"
        assert len(self.org_solar_files) == len(self.mask_solar_files), "Length mismatch between images and masks!"
        if mask is not None:
            self.add_mask(mask)

    def add_mask(self, mask: List[bool]) -> None:
        """Filters out files and masks not required for the current dataset split.add()

        Args:
            mask (List[bool]): list of bollean values, one for each tile, to decide whether to keep it or not.
        """
        assert len(mask) == len(self.org_solar_files), \
            f"Mask is the wrong size! Expected {len(self.org_solar_files)}, got {len(mask)}"
        self.org_solar_files = [x for include, x in zip(mask, self.org_solar_files) if include]
        self.mask_solar_files = [x for include, x in zip(mask, self.mask_solar_files) if include]

    def __len__(self) -> int:
        return len(self.org_solar_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = tif.imread(self.org_solar_files[index]).transpose(1, 2, 0)
        y = tif.imread(self.mask_solar_files[index])
        if self.transform is not None:
            pair = self.transform(image=x, mask=y)
            x = pair.get("image")
            y = pair.get("mask")
        return x, y


class DydasSegmentationDataset(Dataset):

    def __init__(
        self,
        data_folder: Path,
        transform: Callable = True,
    ) -> None:
        self.transform = transform
        # find images and masks inside the specified folder
        self.solar_files = sorted(list(data_folder.glob("*_rgbir.tif")))
        self.mask_files = sorted(list(data_folder.glob("*_bin.tif")))
        # check consistency
        assert len(self.solar_files) == len(self.mask_files), "Images and masks mismatch!"
        for img_path, msk_path in zip(self.solar_files, self.mask_files):
            assert img_path.stem.replace("_rgbir", "") == msk_path.stem.replace("_bin", ""), \
                f"Image and mask mismatch: '{img_path.stem}' - '{msk_path.stem}'"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = tif.imread(self.solar_files[index]).astype(np.float32)[:3]
        y = tif.imread(self.mask_files[index]).astype(np.float32) / 255.0
        if self.transform is not None:
            pair = self.transform(image=x, mask=y)
            x = pair.get("image")
            y = pair.get("mask")
        return x, y

    def __len__(self) -> int:
        return len(self.solar_files)
