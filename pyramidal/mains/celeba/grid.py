import torchvision
import pyramidal.utils.transforms as transforms
from pyramidal.mains.celeba.common import CelebAMain
from pyramidal.autoencoders.densenet.pyramidal457101215 import Pyramidal457101215DenseNetAutoencoder
from pyramidal.mains.celeba.hparams import celeba_hparams as hparams


if __name__ == '__main__':
    """Run this script for training an autoencoder with a grid of blocks"""

    # model
    model = Pyramidal457101215DenseNetAutoencoder(48, 16)

    # size is 64 or 128, depending on hparams settings
    size = hparams['crop_size'][hparams['dataset']]

    # todo: 128 not implemented yet
    random_squares = {
        64: [
            # (size, x, y)
            [(12, 11, 11), (12, 25, 11), (12, 39, 11), (12, 11, 25), (12, 25, 25), (12, 39, 25), (12, 11, 39), (12, 25, 39), (12, 39, 39)],
            [(12, 13, 11), (12, 27, 11), (12, 41, 11), (12, 13, 25), (12, 27, 25), (12, 41, 25), (12, 13, 39), (12, 27, 39), (12, 41, 39)],
            [(12, 11, 13), (12, 25, 13), (12, 39, 13), (12, 11, 27), (12, 25, 27), (12, 39, 27), (12, 11, 41), (12, 25, 41), (12, 39, 41)],
            [(12, 13, 13), (12, 27, 13), (12, 41, 13), (12, 13, 27), (12, 27, 27), (12, 41, 27), (12, 13, 41), (12, 27, 41), (12, 41, 41)],
        ]
    }

    # todo: 128 not implemented yet
    squares = {
        64: [
            # (size, x, y)
            [(12, 12, 12), (12, 26, 12), (12, 40, 12), (12, 12, 26), (12, 26, 26), (12, 40, 26), (12, 12, 40), (12, 26, 40), (12, 40, 40)],
        ]
    }

    # damage transformations
    X_train_transform = torchvision.transforms.Compose([
        transforms.AddSquares(random_squares[size]),
    ])

    X_val_transform = torchvision.transforms.Compose([
        transforms.AddSquares(squares[size]),
    ])

    # train model
    main = CelebAMain('grid', X_train_transform, X_val_transform, model)
    main.run()