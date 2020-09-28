import torchvision
import numpy as np
import pyramidal.utils.transforms as transforms
from pyramidal.mains.celeba.common import CelebAMain
from pyramidal.autoencoders.densenet.pyramidal457101215 import Pyramidal457101215DenseNetAutoencoder
from pyramidal.mains.celeba.hparams import celeba_hparams as hparams



if __name__ == '__main__':
    """Run this script for training an autoencoder with noise"""

    # model
    model = Pyramidal457101215DenseNetAutoencoder(48, 16)

    # size is 64 or 128, depending on hparams settings
    size = hparams['crop_size'][hparams['dataset']]

    mean = 0.
    std = 125.

    # damage transformations
    X_train_transform = torchvision.transforms.Compose([
        transforms.AddRandomGaussianNoise(mean=mean, std=std),
    ])

    X_val_transform = torchvision.transforms.Compose([
        transforms.AddGaussianNoise([
            np.random.normal(mean, std, [size, size, 3]),
            np.random.normal(mean, std, [size, size, 3]),
            np.random.normal(mean, std, [size, size, 3]),
        ]),
    ])

    # train model
    main = CelebAMain('noise', X_train_transform, X_val_transform, model)
    main.run()