import torchvision
import pyramidal.utils.transforms as transforms
from pyramidal.mains.celeba.common import CelebAMain
from pyramidal.autoencoders.densenet.pyramidal457101215 import Pyramidal457101215DenseNetAutoencoder
from pyramidal.mains.celeba.hparams import celeba_hparams as hparams


if __name__ == '__main__':
    """Run this script for training an autoencoder with face blocks"""

    # model
    model = Pyramidal457101215DenseNetAutoencoder(48, 16)

    # size is 64 or 128, depending on hparams settings
    size = hparams['crop_size'][hparams['dataset']]

    random_squares = {
        64: [
            # (size, x, y)
            [(32, 15, 15)],
            [(32, 15, 17)],
            [(32, 17, 15)],
            [(32, 17, 17)],
        ],
        128: [
            [(64, 30, 30)],
            [(64, 30, 34)],
            [(64, 34, 30)],
            [(64, 34, 34)],
        ]
    }

    squares = {
        64: [
            # (size, x, y)
            [(32, 16, 16)],
        ],
        128: [
            [(64, 32, 32)],
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
    main = CelebAMain('block', X_train_transform, X_val_transform, model)
    main.run()