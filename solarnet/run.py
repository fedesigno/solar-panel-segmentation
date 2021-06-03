import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score
from matplotlib import pyplot as plt

from solarnet.preprocessing import MaskMaker, ImageSplitter
from solarnet.datasets import USGSClassifierDataset, USGSSegmentationDataset, DydasSegmentationDataset, make_masks
from solarnet.datasets.transforms import train_transforms, test_transforms
from solarnet.models import Classifier, Segmenter, train_classifier, train_segmenter
from solarnet.logging.tensorboard import TensorBoardLogger
from solarnet.utils import current_timestamp

LOG = logging.getLogger(__name__)


class RunTask:

    @staticmethod
    def make_masks(data_folder='data'):
        """Saves masks for each .tif image in the raw dataset. Masks are saved
        in  <org_folder>_mask/<org_filename>.npy where <org_folder> should be the
        city name, as defined in `data/README.md`.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        """
        mask_maker = MaskMaker(data_folder=Path(data_folder))
        mask_maker.process()

    @staticmethod
    def split_images(data_folder='data', imsize=256, empty_ratio=2):
        """Generates images (and their corresponding masks) of height = width = imsize
        for input into the models.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        imsize: int, default: 224
            The size of the images to be generated
        empty_ratio: int, default: 2
            The ratio of images without solar panels to images with solar panels.
            Because images without solar panels are randomly sampled with limited
            patience, having this number slightly > 1 yields a roughly 1:1 ratio.
        """
        splitter = ImageSplitter(data_folder=Path(data_folder))
        splitter.process(imsize=imsize, empty_ratio=empty_ratio)

    @staticmethod
    def train_classifier(max_epochs=100,
                         warmup=2,
                         patience=5,
                         val_size=0.1,
                         test_size=0.2,
                         data_folder='data',
                         pretrained=True,
                         backbone='resnet50',
                         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the classifier

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        pretrained: bool
            Whether to use pretrained weights (ImageNet) or not
        backbone: str
            Which ResNet to use, among resnet34, resnet50 and resnet101 (defaults to 50)
        device: torch.device, default: cuda if available, else cpu
            The device to train the models on
        """
        data_folder = Path(data_folder)

        model = Classifier(imagenet_base=pretrained, backbone=backbone)
        if device.type != 'cpu':
            model = model.cuda()

        processed_folder = data_folder / 'processed'
        dataset = USGSClassifierDataset(data_folder=processed_folder)

        # make a train and val set
        train_mask, val_mask, test_mask = make_masks(len(dataset), val_size, test_size)

        dataset.add_mask(train_mask)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(USGSClassifierDataset(mask=val_mask,
                                                          data_folder=processed_folder,
                                                          transform_images=False),
                                    batch_size=64,
                                    shuffle=True)
        test_dataloader = DataLoader(USGSClassifierDataset(mask=test_mask,
                                                           data_folder=processed_folder,
                                                           transform_images=False),
                                     batch_size=64)

        train_classifier(model,
                         train_dataloader,
                         val_dataloader,
                         max_epochs=max_epochs,
                         warmup=warmup,
                         patience=patience)

        savedir = data_folder / 'models'
        if not savedir.exists():
            savedir.mkdir()
        torch.save(model.state_dict(), savedir / 'classifier.model')

        # save predictions for analysis
        print("Generating test results")
        preds, true = [], []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_preds = model(test_x)
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())

        np.save(savedir / 'classifier_preds.npy', np.concatenate(preds))
        np.save(savedir / 'classifier_true.npy', np.concatenate(true))

    @staticmethod
    def train_segmenter(max_epochs: int = 100,
                        val_size: float = 0.1,
                        test_size: float = 0.2,
                        lr: float = 1e-3,
                        patience: int = 5,
                        batch_size: int = 64,
                        logs_folder: str = "logs",
                        data_folder: str = "data",
                        model_folder: str = "models",
                        pretrained_encoder: str = None,
                        device: str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the segmentation model

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        use_classifier: boolean, default: True
            Whether to use the pretrained classifier (saved in data/models/classifier.model by the
            train_classifier step) as the weights for the downsampling step of the segmentation
            model
        device: torch.device, default: cuda if available, else cpu
            The device to train the models on
        """
        experiment_id = current_timestamp()
        data_folder = Path(data_folder)
        model_folder = Path(model_folder) / experiment_id
        if not model_folder.exists():
            model_folder.mkdir()
        LOG.info("Starting experiment: %s", experiment_id)

        model = smp.Unet(encoder_weights="imagenet", in_channels=3, classes=1)
        model = model.to(device=device)

        if pretrained_encoder is not None:
            pretrained_encoder = Path(pretrained_encoder)
            if not pretrained_encoder.exists():
                raise ValueError(f"The specified path to the pretrained weights does not exist: {pretrained_encoder}")
            encoder_state = torch.load(pretrained_encoder)
            model.load_base(encoder_state)

        logger = TensorBoardLogger(log_folder=Path(logs_folder) / experiment_id)
        dataset = USGSSegmentationDataset(data_folder=data_folder, transform=train_transforms())
        train_mask, val_mask, test_mask = make_masks(len(dataset), val_size, test_size)

        dataset.add_mask(train_mask)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_dataloader = DataLoader(USGSSegmentationDataset(mask=val_mask,
                                                            data_folder=data_folder,
                                                            transform=test_transforms()),
                                    batch_size=batch_size,
                                    num_workers=8,
                                    pin_memory=True)
        test_dataloader = DataLoader(USGSSegmentationDataset(mask=test_mask,
                                                             data_folder=data_folder,
                                                             transform=test_transforms()),
                                     batch_size=batch_size,
                                     num_workers=8,
                                     pin_memory=True)
        loss = nn.BCEWithLogitsLoss().to(device)
        optimizer = Adam(params=model.parameters(), lr=lr)
        scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=max_epochs)
        train_segmenter(model,
                        criterion=loss,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        device=device,
                        logger=logger,
                        max_epochs=max_epochs,
                        patience=patience)

        torch.save(model.state_dict(), model_folder / 'segmenter.pth')

        LOG.info("Generating test results")
        images, preds, true = [], [], []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_x = test_x.to(device, non_blocking=True)
                test_y = test_y.to(device, non_blocking=True)
                test_preds = model(test_x)
                images.append(test_x.cpu().numpy())
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())

        np.save(model_folder / 'segmenter_images.npy', np.concatenate(images))
        np.save(model_folder / 'segmenter_pred.npy', np.concatenate(preds))
        np.save(model_folder / 'segmenter_true.npy', np.concatenate(true))

    def train_both(self,
                   c_max_epochs=100,
                   c_warmup=2,
                   c_patience=5,
                   c_val_size=0.1,
                   c_test_size=0.2,
                   s_max_epochs=100,
                   s_warmup=2,
                   s_patience=5,
                   s_val_size=0.1,
                   s_test_size=0.2,
                   data_folder='data',
                   pretrained=True,
                   backbone='resnet50',
                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the classifier, and use it to train the segmentation model.
        """
        data_folder = Path(data_folder)
        self.train_classifier(max_epochs=c_max_epochs,
                              val_size=c_val_size,
                              test_size=c_test_size,
                              warmup=c_warmup,
                              patience=c_patience,
                              data_folder=data_folder,
                              pretrained=pretrained,
                              backbone=backbone,
                              device=device)
        self.train_segmenter(max_epochs=s_max_epochs,
                             val_size=s_val_size,
                             test_size=s_test_size,
                             warmup=s_warmup,
                             patience=s_patience,
                             use_classifier=True,
                             data_folder=data_folder,
                             device=device)

    @staticmethod
    def test_segmenter(data_folder="data", results_folder="data/results", store_predictions: bool = True):
        data_folder = Path(data_folder)
        results_folder = Path(results_folder)

        images = np.load(data_folder / "segmenter_images.npy")
        preds = np.load(data_folder / "segmenter_preds.npy")
        target = np.load(data_folder / "segmenter_true.npy")
        metrics_fn = OrderedDict([("f1", f1_score), ("iou", jaccard_score), ("accuracy", accuracy_score),
                                  ("precision", precision_score), ("recall", recall_score)])

        num_images = images.shape[0]
        num_metrics = len(metrics_fn)

        metrics = np.zeros((num_images, num_metrics))
        preds = (preds > 0.5).astype(np.int)  # a bit arbitrary, but should make sense
        target = target.astype(np.int)

        for i in tqdm(range(num_images)):
            y_true = target[i].flatten()
            y_pred = preds[i].flatten()
            for j, score_fn in enumerate(metrics_fn.values()):
                metrics[i, j] = score_fn(y_true, y_pred)
            if store_predictions:
                plt.imsave(results_folder / f"{i:06d}_true.png", target[i])
                plt.imsave(results_folder / f"{i:06d}_pred.png", preds[i])

        mean_metrics = metrics.mean(axis=0)
        for i, name in enumerate(metrics_fn.keys()):
            print(f"{name:<20s}: {mean_metrics[i]:.4f}")

    @staticmethod
    def test_finetuning(data_folder="data",
                        model_folder="data/models",
                        results_folder="data/custom",
                        store_predictions: bool = False,
                        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        data_folder = Path(data_folder)
        model_folder = Path(model_folder)

        model = Segmenter()
        checkpoint = torch.load(model_folder / "segmenter.model")
        model.load_state_dict(checkpoint)

        if device.type != 'cpu':
            model = model.cuda()
        test_dataloader = DataLoader(DydasSegmentationDataset(data_folder=data_folder, transform_images=False),
                                     batch_size=64,
                                     shuffle=False)

        true = []
        preds = []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_preds = model(test_x)
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())

        metrics_fn = OrderedDict([("f1", f1_score), ("iou", jaccard_score), ("accuracy", accuracy_score),
                                  ("precision", precision_score), ("recall", recall_score)])

        num_images = len(test_dataloader)
        num_metrics = len(metrics_fn)
        metrics = np.zeros((num_images, num_metrics))

        for i in tqdm(range(num_images)):
            y_true = true[i].flatten().astype(np.int)
            y_pred = (preds[i].flatten() > 0.5).astype(np.int)

            for j, score_fn in enumerate(metrics_fn.values()):
                metrics[i, j] = score_fn(y_true, y_pred)
            if store_predictions:
                plt.imsave(results_folder / f"{i:06d}_true.png", true[i])
                plt.imsave(results_folder / f"{i:06d}_pred.png", preds[i])

        mean_metrics = metrics.mean(axis=0)
        for i, name in enumerate(metrics_fn.keys()):
            print(f"{name:<20s}: {mean_metrics[i]:.4f}")
