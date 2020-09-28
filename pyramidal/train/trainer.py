import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import math
import os.path as path


class Trainer():
    """This class implements the core work of the model training."""

    def __init__(self, train_loader, val_loader, test_loader, model, optimizer, criterion, hparams, log_dir, model_dir):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = hparams['device']
        self.log_interval = hparams['log_interval']
        self.num_epochs = hparams['num_epochs']
        self.batch_size = hparams['batch_size'][hparams['gpu']]
        self.num_images_written = min(8, self.batch_size)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.model = model.to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0 or batch_idx >= len(self.train_loader):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                     100. * batch_idx / len(self.train_loader), loss.item()))
        return np.mean(losses)

    def eval_epoch(self):
        self.model.eval()
        eval_losses = []
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                eval_losses.append(self.criterion(output, target).item())
        eval_loss = np.mean(eval_losses)
        print('Eval set: Average loss: {:.4f}'.format(eval_loss))
        return eval_loss

    def train(self):
        # recording will be done in tensorboard
        with SummaryWriter(self.log_dir) as writer:

            try:
                best_te_loss_db = 0

                # loop of epochs
                for epoch in range(1, self.num_epochs + 1):

                    # train and eval
                    tr_loss = self.train_epoch(epoch)
                    te_loss = self.eval_epoch()

                    # record losses as PSNR(dB)
                    # it is expected that the input and the output tensors of the model range from -1 to +1.
                    # losses are calculated in that space.
                    # so, if we want to calculate the equivalent PSNR(db) as it were in a range from 0 to +1,
                    # 6 dB must be added.
                    tr_loss_db = 6 - 10 * math.log10(tr_loss)
                    te_loss_db = 6 - 10 * math.log10(te_loss)
                    writer.add_scalars('PSNR (dB)', {
                        'train': tr_loss_db,
                        'validation': te_loss_db,
                    }, epoch)

                    # save best model
                    if te_loss_db > best_te_loss_db:
                        torch.save(self.model.state_dict, path.join(self.model_dir, 'best_model.pt'))
                        best_te_loss_db = te_loss_db

                    # record images.
                    # the output tensors of the model range from -1 to +1: denormalize to 0 to +1 before recording the images
                    # note: model is already in eval after eval_epoch above
                    with torch.no_grad():
                        # test images
                        bX, btarget = self.write_images(writer, self.test_loader, '1. Output (test)', epoch)

                        # input image and ground truth (only first time)
                        if epoch == 1:
                            writer.add_images('2. Input', Trainer.denormalize(bX), epoch)
                            writer.add_images('3. Ground truth', Trainer.denormalize(btarget), epoch)

                        # train images
                        self.write_images(writer, self.train_loader, '4. Output (train)', epoch)

            except KeyboardInterrupt:
                print('-' * 89)
                print('Exiting from training early')

    @staticmethod
    def denormalize(img):
        return img * 0.5 + 0.5

    def write_images(self, writer, loader, title, epoch):
        iter_ = iter(loader)
        bX, btarget = next(iter_)
        bX = bX[0:min(8, self.batch_size), :, :, :]
        btarget = btarget[0:self.num_images_written, :, :, :]
        output = self.model(bX.to(self.device))
        writer.add_images(title, Trainer.denormalize(output), epoch)
        return bX, btarget
