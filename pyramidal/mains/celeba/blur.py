import torchvision
import pyramidal.utils.transforms as transforms
from pyramidal.mains.celeba.common import CelebAMain
from pyramidal.autoencoders.densenet.pyramidal457101215 import Pyramidal457101215DenseNetAutoencoder

if __name__ == '__main__':
    """Run this script for training an autoencoder with blur"""

    # model
    model = Pyramidal457101215DenseNetAutoencoder(48, 16)

    # damage transformations
    X_train_transform = torchvision.transforms.Compose([
        transforms.Blur((15, 15)),
    ])

    X_val_transform = torchvision.transforms.Compose([
        transforms.Blur((15, 15)),
    ])

    # train model
    main = CelebAMain('blur', X_train_transform, X_val_transform, model)
    main.run()