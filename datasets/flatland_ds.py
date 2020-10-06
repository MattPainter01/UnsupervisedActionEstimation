import random

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.transforms import PairTransform


class ForwardVAEDS(Dataset):
    def __init__(self, path_input, path_action, transforms=None, output_targets=True, mean_channels=False, num_steps=1, noise_name=None):
        """ FlatLand Dataset

        Args:
            path_input (str): Root path to images.npy
            path_action (str): Root path to actions.npy
            transforms (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            output_targets (bool): If True output image pair corresponding to symmetry action. If False, standard dSprites.
            mean_channels (bool): If True mean over the 3 image channels. If False returns image with 3 colour channels
            num_steps (int): Number of steps/actions to apply
            noise_name (str): Name of noise to add, default None
        """
        self.inputs = np.load(path_input).reshape(-1, 64, 64, 3).transpose((0, 3, 1, 2))
        self.actions = np.load(path_action)
        self.actions = np.array(self.actions).reshape(-1, 1)
        self.transforms = transforms
        self.output_targets = output_targets
        self.mean_channels = mean_channels
        self.num_steps = num_steps
        self.factors = None
        self.latent_to_index_dict = None
        self.noise_transform = PairTransform(noise_name) if noise_name is not None else None

        self.get_factors()

        self.possible_latents = np.arange(17, 49)
        self.latents_sizes = (len(self.possible_latents), len(self.possible_latents))

    def __getitem__(self, index):
        input_batch = self.inputs[index]
        action_batch = self.actions[index]
        if self.num_steps != -1:
            target_batch = self.inputs[index+self.num_steps]
        else:
            tindex = random.randint(0, len(self)-1)
            target_batch = self.inputs[tindex]

        if self.transforms is not None:
            input_batch = self.transforms(input_batch)
            target_batch = self.transforms(target_batch)
            action_batch = self.transforms(action_batch).long()

        if self.mean_channels:
            input_batch = input_batch.mean(0, keepdim=True)
            target_batch = target_batch.mean(0, keepdim=True)

        if self.noise_transform is not None:
            input_batch, target_batch = self.noise_transform(input_batch, target_batch)

        if self.output_targets:
            out = (input_batch, action_batch), target_batch
        else:
            out = input_batch, action_batch
        return out

    def __len__(self):
        if self.num_steps != -1:
            count = len(self.inputs) - self.num_steps - 1
        else:
            count = len(self.inputs)
        return count

    def get_factors(self):
        factors = []
        latent_to_index = {}
        for i in range(len(self)):
            img = self.__getitem__(i)[0]
            if self.output_targets:
                img = img[0]

            h = torch.argmax(img.mean(1)).item()
            w = torch.argmax(img.mean(2)).item()
            factors.append((h, w))
            latent_to_index[(h, w)] = i
        self.factors = factors
        self.latent_to_index_dict = latent_to_index

    def generative_factors(self, index):
        return self.index_to_latent(index)

    def latent_to_index(self, latents):
        x, y = latents[0], latents[1]

        try:
            x, y = x.item(), y.item()
        except:
            pass

        if (x, y) not in self.factors:
            for i in range(-6, 6):
                for j in range(-6, 6):
                    if (x+i, y+j) in self.factors:
                        return self.latent_to_index_dict[(x+i, y+j)]

        return self.latent_to_index_dict[(x, y)]

    def index_to_latent(self, index):
        return self.factors[index]

    def get_img_by_latent(self, latent):
        index = self.latent_to_index(latent)
        if self.output_targets:
            return self.__getitem__(index)[0]
        else:
            return self.__getitem__(index)

    def sample_latent(self):
        return list(list(self.latent_to_index_dict.keys())[np.random.randint(0, len(self.latent_to_index_dict))])

