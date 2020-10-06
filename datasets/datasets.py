import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Lambda, ToTensor

from datasets.flatland_ds import ForwardVAEDS
from datasets.dsprites import PairSprites


def sprites_transforms(_):
    return ToTensor(), ToTensor()


def forward_ds_transforms(_):
    lam = lambda x: torch.from_numpy(np.array(x)).float()
    return Lambda(lam), Lambda(lam)


transforms = {
    'flatland': forward_ds_transforms,
    'dsprites': sprites_transforms,
}


def split(func):  # Splits a dataset into a train and val set
    def splitter(args):
        ds = func(args)
        lengths = int(len(ds) * (1 - args.split)), int(len(ds)) - int(len(ds) * (1 - args.split))
        train_ds, val_ds = random_split(ds, lengths) if args.split > 0 else (ds, ds)
        return train_ds, val_ds

    return splitter


def fix_data_path(func):  # Sets the datapath to that in _default_paths if it is None
    def fixer(args):
        args.data_path = args.data_path if args.data_path is not None else _default_paths[args.dataset]
        return func(args)

    return fixer


def set_to_loader(trainds, valds, args):
    trainloader = DataLoader(trainds, batch_size=args.batch_size, num_workers=7, shuffle=args.shuffle, drop_last=False,
                             pin_memory=True)
    valloader = DataLoader(valds, batch_size=args.batch_size, num_workers=7, shuffle=False, drop_last=False,
                           pin_memory=True)
    return trainloader, valloader


@split
@fix_data_path
def sprites(args):
    train_transform, test_transform = transforms[args.dataset](args)
    output_targets = True if args.model in ['forward', 'rgrvae'] else False
    ds = PairSprites(args.data_path, download=False, transform=train_transform, wrapping=True, offset=args.offset,
                     noise_name=args.noise_name, output_targets=output_targets)
    return ds


@split
@fix_data_path
def forward_vae_ds(args):
    import os
    train_transform, test_transform = transforms[args.dataset](args)
    output_targets = True if args.model in ['forward', 'rgrvae'] else False
    mean_channels = True

    images_path = os.path.join(args.data_path, 'inputs.npy')
    actions_path = os.path.join(args.data_path, 'actions.npy')
    ds = ForwardVAEDS(images_path, actions_path, transforms=train_transform, output_targets=output_targets,
                      mean_channels=mean_channels, num_steps=args.offset, noise_name=args.noise_name)
    return ds


_default_paths = {
    'flatland': '/home/matt/PycharmProjects/PyTorchCode/NeurIPS19-SBDRL/code/learn_4_dim_linear_disentangled_representation/flatland/flat_game',
    'dsprites': '',
}

datasets = {
    'flatland': forward_vae_ds,
    'dsprites': sprites,
}

dataset_meta = {
    'flatland': {'nc': 1, 'factors': 2, 'max_classes': 40},
    'dsprites': {'nc': 1, 'factors': 5, 'max_classes': 40},
}
