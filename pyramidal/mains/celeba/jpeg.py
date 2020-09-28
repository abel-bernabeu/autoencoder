import torchvision
import pyramidal.utils.transforms as transforms
from pyramidal.mains.celeba.common import CelebAMain
from pyramidal.autoencoders.densenet.pyramidal457101215 import Pyramidal457101215DenseNetAutoencoder
from pyramidal.mains.celeba.hparams import celeba_hparams as hparams


if __name__ == '__main__':
    """Run this script for training an autoencoder with jpeg artifacts"""

    # model
    model = Pyramidal457101215DenseNetAutoencoder(48, 16)

    # size is 64 or 128, depending on hparams settings
    size = hparams['crop_size'][hparams['dataset']]

    # quality for jpeg
    quality = 5

    # damage transformations
    X_train_transform = torchvision.transforms.Compose([
        transforms.JPEG(quality=quality),
    ])

    X_val_transform = torchvision.transforms.Compose([
        transforms.JPEG(quality=quality),
    ])

    # train model
    main = CelebAMain('jpeg', X_train_transform, X_val_transform, model)
    main.run()