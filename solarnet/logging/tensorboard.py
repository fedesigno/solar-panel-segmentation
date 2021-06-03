import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from solarnet.logging import BaseLogger


class TensorBoardLogger(BaseLogger):

    def __init__(self, log_folder: Path = Path("logs"), filename_suffix: str = "", current_step: int = 0) -> None:
        super().__init__()
        self.log = SummaryWriter(log_dir=log_folder, filename_suffix=filename_suffix)
        self.current_step = current_step

    def step(self, iteration: int = None) -> None:
        if not iteration:
            self.current_step += 1
        else:
            self.current_step = iteration

    def log_scalar(self, name: str, value: float, **kwargs) -> None:
        self.log.add_scalar(name, value, global_step=self.current_step, **kwargs)

    def log_image(self, name: str, image: np.ndarray, **kwargs) -> None:
        self.log.add_image(name, image, global_step=self.current_step, **kwargs)
