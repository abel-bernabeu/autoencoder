from pyramidal.train.image_folder import ImageFolderTrainer
import os
from pyramidal.tools.celeba import CelebA
from pathlib import Path


class CelebATrainer(ImageFolderTrainer):
    """Subclass of ImageFolderTrainer for the particular case of CelebA dataset. CelebA dataset is downloaded into a folder
    and then proceed as base ImageFolderTrainer."""

    def __init__(self, celeba_root, X_train_transform, X_val_transform, hparams, model, optimizer, criterion, log_dir, model_dir):
        # make CelebA dataset images available at celeba_image_folder
        celeba_image_folder = os.path.join(celeba_root, 'celeba', 'img_align_celeba')
        if not os.path.exists(celeba_image_folder):
            # celeba_image_folder is missing --> donwload CelebA
            Path(celeba_root).mkdir(parents=True, exist_ok=True)
            CelebA(celeba_root, download=True, transform=None)

        if not os.path.exists(model_dir):
            Path(model_dir).mkdir(parents=True, exist_ok=True)

        super().__init__(celeba_image_folder, X_train_transform, X_val_transform, hparams, model, optimizer, criterion, log_dir, model_dir)