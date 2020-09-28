import torch.nn as nn
import torch.optim as optim
import os
import datetime
from pyramidal.train.celeba import CelebATrainer
from pyramidal.mains.celeba.hparams import celeba_hparams as hparams


class CelebAMain():
    """This code is common to all scripts inside pyramidal/mains. The transforms for damaging the images control the type of model generated: denoiser, deblurrer, etc"""

    def __init__(self, test_id, X_train_transform, X_val_transform, model):

        # X (train): noise, jpeg artifacts, B&W, etc
        self.X_train_transform = X_train_transform

        # X (val, test): noise, jpeg artifacts, B&W, etc
        self.X_val_transform = X_val_transform

        # celeba root folder
        self.celeba_root = os.path.join(os.path.expanduser('~'), 'autoencoder', 'datasets', 'celeba')

        # datetime string
        dt = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        # log dir for tensorboard
        self.log_dir = os.path.join(os.path.expanduser('~'), 'autoencoder', 'tensorboard', test_id, dt)

        # dir for saving the model
        self.model_dir = os.path.join(os.path.expanduser('~'), 'autoencoder', 'model', test_id, dt)

        # model
        self.model = model

        # optimizer & criterion
        self.optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
        self.criterion = nn.MSELoss()

    def run(self):
        trainer = CelebATrainer(
            self.celeba_root,
            self.X_train_transform,
            self.X_val_transform,
            hparams,
            self.model,
            self.optimizer,
            self.criterion,
            self.log_dir,
            self.model_dir
        )
        trainer.train()