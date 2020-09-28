import torchvision
import pyramidal.utils.transforms as transforms
from pyramidal.mains.celeba.common import CelebAMain
from pyramidal.autoencoders.densenet.pyramidal457101215 import Pyramidal457101215DenseNetAutoencoder
from pyramidal.mains.celeba.hparams import celeba_hparams as hparams


if __name__ == '__main__':
    """Run this script for training an autoencoder with pixelation"""

    # model
    model = Pyramidal457101215DenseNetAutoencoder(48, 16)

    # size is 64 or 128, depending on hparams settings
    size = hparams['crop_size'][hparams['dataset']]

    # super resolution scale factor
    scale = 8

    # damage transformations
    X_train_transform = torchvision.transforms.Compose([
        transforms.Pixelate(scale=scale),
    ])

    X_val_transform = torchvision.transforms.Compose([
        transforms.Pixelate(scale=scale),
    ])

    # train model
    main = CelebAMain('pixelate', X_train_transform, X_val_transform, model)
    main.run()