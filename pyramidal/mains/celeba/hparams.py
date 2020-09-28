import torch

"""This settings are common to all scripts in pyramidal/mains"""

celeba_hparams = {
    # gpu =
    #    small if VRAM <= 4 GB
    #    large if VRAM > 4 GB
    'gpu': 'small' if torch.cuda.get_device_properties(0).total_memory <= 4294967296 else 'large',

    # PROGRAMMABLE: depends on gpu setting above
    'batch_size': {
        'small': 10,
        'large': 80,
    },

    'max_dataset_size': {
        'small': 1600,
        'large': 112000,
    },

    # PROGRAMMABLE: controls resize y size below
    'dataset': 'small',

    # for image resizing
    'resize': {
        'small': (87, 71),
        'large': (174, 142),
    },

    # for image cropping: this is the final image size
    'crop_size': {
        'small': 64,
        'large': 128,
    },

    'device': 'cuda',
    'log_interval': 2,
    'num_epochs': 100000,
    'num_workers': 4,

    # learning rate and weight decay for adam optimizer
    'lr': 3e-4,
    'weight_decay': 1e-4,
}
