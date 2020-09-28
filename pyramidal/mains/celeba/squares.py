import torchvision
import pyramidal.utils.transforms as transforms
from pyramidal.mains.celeba.common import CelebAMain
from pyramidal.autoencoders.densenet.pyramidal457101215 import Pyramidal457101215DenseNetAutoencoder
from pyramidal.mains.celeba.hparams import celeba_hparams as hparams


if __name__ == '__main__':
    """Run this script for training an autoencoder with small blocks"""

    # model
    model = Pyramidal457101215DenseNetAutoencoder(48, 16)

    # size is 64 or 128, depending on hparams settings
    size = hparams['crop_size'][hparams['dataset']]

    # todo: 128 not implemented yet
    min_size = {
        64: 4,
    }

    # todo: 128 not implemented yet
    max_size = {
        64: 14,
    }

    # todo: 128 not implemented yet
    squares = {
        64: [
            # (size, x, y)
            [(5, 9, 17), (12, 39, 20), (9, 48, 42), (7, 18, 52)],
            [(5, 49, 42), (12, 12, 31), (9, 7, 13), (7, 39, 6)],
            [(5, 49, 17), (12, 12, 20), (9, 7, 42), (7, 39, 52)],
            [(5, 9, 52), (12, 39, 31), (9, 48, 13), (7, 18, 6)],
        ]
    }

    # damage transformations
    X_train_transform = torchvision.transforms.Compose([
        transforms.AddRandomSquares(min_n=3, max_n=4, min_size=min_size[size], max_size=max_size[size]),
    ])

    X_val_transform = torchvision.transforms.Compose([
        transforms.AddSquares(squares[size]),
    ])

    # train model
    main = CelebAMain('squares', X_train_transform, X_val_transform, model)
    main.run()