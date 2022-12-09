import numpy as np
import torchvision.io
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from torchvision.io import ImageReadMode

from one_hot import OneHotEncoder


class CoalsDataset(Dataset):
    def __init__(self, root: str, colors: Tensor, class_names, transform=None):
        self.root = root
        self.colors = colors
        self.class_names = class_names
        self.transform = transform
        self.data_list = np.load(f'{root}/index.npy')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index) -> T_co:
        names = self.data_list[index]
        image_file_path = f'{self.root}/{names[0]}'
        mask_file_path = f'{self.root}/{names[1]}'
        image = torchvision.io.read_image(image_file_path, ImageReadMode.RGB)
        mask = torchvision.io.read_image(mask_file_path, ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = OneHotEncoder.encode_mask_image(mask, self.colors)
        return image, mask
