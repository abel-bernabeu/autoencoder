import torch
import torchvision
import pyramidal.utils.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import PIL
from pyramidal.train.trainer import Trainer


class ImageFolderTrainer():
    """This class trains an autoencoder model with the images contained inside a folder"""

    def __init__(self, image_folder, X_train_transform, X_val_transform, hparams, model, optimizer, criterion, log_dir, model_dir):
        super().__init__()

        # folder with images for the autoencoder training
        self.image_folder = image_folder

        # X (train): noise, jpeg artifacts, B&W, etc
        self.X_train_transform = X_train_transform

        # X (val, test): noise, jpeg artifacts, B&W, etc
        self.X_val_transform = X_val_transform

        # hparams
        self.hparams = hparams

        self.resize = hparams['resize'][hparams['dataset']]
        self.size = hparams['crop_size'][hparams['dataset']]
        self.batch_size = hparams['batch_size'][hparams['gpu']]
        self.num_workers = hparams['num_workers']
        self.max_size = hparams['max_dataset_size'][hparams['gpu']]

        # model & train
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.log_dir = log_dir
        self.model_dir = model_dir

    def train(self):
        # transformations = pre + X + post

        # pre (train): randomness
        pre_train_transforms = []

        if not self.resize == None:
            pre_train_transforms.append(torchvision.transforms.Resize(self.resize, interpolation=PIL.Image.BICUBIC))

        pre_train_transforms.extend([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
            torchvision.transforms.RandomAffine(0, scale=(0.95, 1.05), resample=PIL.Image.BICUBIC),
            torchvision.transforms.CenterCrop(self.size),
        ])

        pre_train_transform = torchvision.transforms.Compose(pre_train_transforms)

        # pre (val, test)
        pre_val_transforms = []

        if not self.resize == None:
            pre_val_transforms.append(torchvision.transforms.Resize(self.resize, interpolation=PIL.Image.BICUBIC))

        pre_val_transforms.extend([
            torchvision.transforms.CenterCrop(self.size),
        ])

        pre_val_transform = torchvision.transforms.Compose(pre_val_transforms)

        # post: to tensor from -1 to 1
        post_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)
        ])

        # train dataset: the dataset returns (X, Y), where X is an image altered by X_transform above and Y is its corresponding ground truth
        train_dataset = datasets.ImageFolderAutoEncoderDataset(
            self.image_folder,
            'train',
            pre_transform=pre_train_transform,
            X_transform=self.X_train_transform,
            post_transform=post_transform,
            max_size=self.max_size
        )

        # test dataset
        test_dataset = datasets.ImageFolderAutoEncoderDataset(
            self.image_folder,
            'test',
            pre_transform=pre_val_transform,
            X_transform=self.X_val_transform,
            post_transform=post_transform,
            max_size=self.max_size
        )

        # validation dataset
        val_dataset = datasets.ImageFolderAutoEncoderDataset(
            self.image_folder,
            'val',
            pre_transform=pre_val_transform,
            X_transform=self.X_val_transform,
            post_transform=post_transform,
            max_size=self.max_size
        )

        # data loaders: shuffle only train data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # train
        trainer = Trainer(train_loader, val_loader, test_loader, self.model, self.optimizer, self.criterion, self.hparams, self.log_dir, self.model_dir)
        trainer.train()
