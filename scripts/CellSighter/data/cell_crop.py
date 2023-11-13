import cv2
import numpy as np


class CellCrop:
    """
    Represent crop of cell with all the channels, mask of cell, mask cells in the environment and more
    """
    def __init__(self, cell_id, image_id, label, slices, cells, image):
        self._cell_id = cell_id
        self._image_id = image_id
        self._label = label
        self._slices = slices
        self._cells = cells
        self._image = image

    def sample(self, mask=False):
        result = {'cell_id': self._cell_id, 'image_id': self._image_id,
                  'image': self._image[self._slices].astype(np.float32),
                  'slice_x_start': self._slices[0].start,
                  'slice_y_start': self._slices[1].start,
                  'slice_x_end': self._slices[0].stop,
                  'slice_y_end': self._slices[1].stop,
                  'label': self._label.astype(np.long)}
        if mask:
            result['mask'] = (self._cells[self._slices] == self._cell_id).astype(np.float32)
            result['all_cells_mask'] = (self._cells[self._slices] > 0).astype(np.float32)
            result['all_cells_mask_seperate'] = (self._cells[self._slices]).astype(np.float32)

        return result
