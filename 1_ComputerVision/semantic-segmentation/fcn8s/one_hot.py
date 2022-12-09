import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import functional as F


class OneHotEncoder():
    @staticmethod
    def encode_mask_image_1(mask_image: ndarray, colors: ndarray) -> ndarray:
        height, width = mask_image.shape[1:]
        one_hot_mask = np.zeros((len(colors), height, width), dtype=np.float32)
        for label_index, label in enumerate(colors):
            one_hot_mask[label_index, :, :] = np.all(mask_image == label[:, None, None], axis=0).astype(float)
        return one_hot_mask

    @staticmethod
    def encode_mask_image(mask_image: Tensor, colors: Tensor) -> Tensor:
        height, width = mask_image.shape[1:]
        one_hot_mask = torch.zeros([len(colors), height, width], dtype=torch.float)
        for label_index, label in enumerate(colors):
            one_hot_mask[label_index, :, :] = torch.all(mask_image == label[:, None, None], dim=0).float()
        return one_hot_mask

    @staticmethod
    def encode_score(score: Tensor) -> Tensor:
        num_classes = score.shape[1]
        label = torch.argmax(score, dim=1)
        pred = F.one_hot(label, num_classes=num_classes)
        return pred.permute(0, 3, 1, 2)

    @staticmethod
    def encode_label(label: Tensor, n_classes) -> Tensor:
        one_hot_label = F.one_hot(label, num_classes=n_classes)
        return one_hot_label.permute(0, 3, 1, 2).float()

    @staticmethod
    def decode(input: Tensor, colors: Tensor):
        height, width = input.shape[1:]
        mask = torch.zeros([3, height, width], dtype=torch.long)
        for label_num in range(0, len(colors)):
            index = (input[label_num] == 1)
            mask[:, index] = colors[label_num][:, None]
        return mask
