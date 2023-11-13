from torchvision.transforms import Lambda, RandomCrop, CenterCrop
import numpy as np
import torch


class ShiftAugmentation(torch.nn.Module):
    """
    Augmentation that shift each marker channel a few pixels in random direction
    """
    def __init__(self, n_size, shift_max=0):
        super(ShiftAugmentation, self).__init__()
        self.shift_max = shift_max
        self.n_size = n_size
        p = 0.3

        self.chanel_shifter = Lambda(lambda x:
                                     RandomCrop(size=n_size)(
                                         CenterCrop(size=n_size + (self.shift_max if np.random.random() < p else 0))(x)))

    def forward(self, x):
        # X is shaped: (C, H, W)
        aug_x = torch.zeros((x.shape[0], self.n_size, self.n_size))
        for i in range(x.shape[0]):
            aug_x[i, :, :] = self.chanel_shifter(x[[i], :, :])[0,:,:]
        return aug_x
