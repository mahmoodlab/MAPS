import cv2
import numpy as np
import torchvision
from torchvision.transforms import Lambda
from data.shift_augmentation import ShiftAugmentation


def poisson_sampling(x):
    """
    Augmentation that resample the data from poisson distribution.
    Args:
        x: (H,W,C) when C is th number of markers + 2. (one for the mask of the cell.
        one for the mask of all the other cells in the environment)

    Returns:
        Augmented tensor size of (H,W,C) resampled from poisson distribution.
    """
    blur = cv2.GaussianBlur(x[:, :, :-2], (5, 5), 0)
    x[:, :, :-2] = np.random.poisson(lam=blur, size=x[:, :, :-2].shape)
    return x


def cell_shape_aug(x):
    """
    Augment the mask of the cell size by dilating the size of the cell with random kernel
    """
    if np.random.random() < 0.5:
        cell_mask = x[:, :, -1]
        kernel_size = np.random.choice([2, 3, 5])
        kernel = np.ones(kernel_size, np.uint8)
        img_dilation = cv2.dilate(cell_mask, kernel, iterations=1)
        x[:, :, -1] = img_dilation
    return x


def env_shape_aug(x):
    """
        Augment the size of the cells mask in the environment,
        by dilating the size of the cell with random kernel
    """
    if np.random.random() < 0.5:
        cell_mask = x[:, :, -2]
        kernel_size = np.random.choice([2, 3, 5])
        kernel = np.ones(kernel_size, np.uint8)
        img_dilation = cv2.dilate(cell_mask, kernel, iterations=1)
        x[:, :, -2] = img_dilation
    return x


val_transform = lambda crop_size: torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.CenterCrop((crop_size, crop_size))
])

train_transform = lambda crop_size, shift: torchvision.transforms.Compose([
    torchvision.transforms.Lambda(poisson_sampling),
    torchvision.transforms.Lambda(cell_shape_aug),
    torchvision.transforms.Lambda(env_shape_aug),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomRotation(degrees=(0, 360)),
    Lambda(lambda x: ShiftAugmentation(shift_max=shift, n_size=crop_size)(x) if np.random.random() < 0.5 else x),
    torchvision.transforms.CenterCrop((crop_size, crop_size)),
    torchvision.transforms.RandomHorizontalFlip(p=0.75),
    torchvision.transforms.RandomVerticalFlip(p=0.75),
])
