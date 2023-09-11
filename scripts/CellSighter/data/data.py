import numpy as np
from torch.utils.data import Dataset


class CellCropsDataset(Dataset):
    def __init__(self,
                 crops,
                 mask=False,
                 transform=None):
        super().__init__()
        self._crops = crops
        self._transform = transform
        self._mask = mask

    def __len__(self):
        return len(self._crops)

    def __getitem__(self, idx):
        sample = self._crops[idx].sample(self._mask)
        aug = self._transform(np.dstack(
            [sample['image'], sample['all_cells_mask'][:, :, np.newaxis], sample['mask'][:, :, np.newaxis]])).float()
        sample['image'] = aug[:-1, :, :]
        sample['mask'] = aug[[-1], :, :]

        return sample

