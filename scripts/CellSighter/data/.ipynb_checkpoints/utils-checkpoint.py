import glob
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
from data.cell_crop import CellCrop
from PIL import Image
from skimage import io


def read_channels(path):
    """
    Reads channels from a line-separated text file.

    Args:
        path (PathLike): A path to a text file of line-separated channels.

    Returns:
        A list of channels.
    """
    with open(path, 'r') as f:
        channels = f.read().strip().split('\n')
    return channels


def filter_channels(channels, blacklist=None):
    """
    Filters out "blacklisted" channels.

    Args:
        channels (List[Tuple[int, str]]): A list of channels.
        blacklist (List[str], optional): A list of blacklisted channel names. Defaults to None.

    Returns:
        List[Tuple[int, str]]: A list of channels.
    """
    return [(i, c) for i, c in enumerate(channels) if c not in blacklist]


def load_data(fname) -> np.ndarray:
    if fname.endswith(".npz"):
        image = np.load(fname, allow_pickle=True)['data']
    elif fname.endswith(".tif") or fname.endswith(".tiff"):
        image = io.imread(fname)
    return image


def load_image(image_path, cells_path, cells2labels_path, channels=[], to_pad=False, crop_size=0):
    image = load_data(image_path)
    if len(channels) > 0:
        image = image[..., channels]
    cells = load_data(cells_path).astype(np.int64)
    if cells2labels_path.endswith(".npz"):
        cells2labels = np.load(cells2labels_path, allow_pickle=True)['data'].astype(np.int32)
    elif cells2labels_path.endswith(".txt"):
        with open(cells2labels_path, "r") as f:
            cells2labels = np.array(f.read().strip().split('\n')).astype(float).astype(int)
    if to_pad:
        image = np.pad(image, ((crop_size // 2, crop_size // 2), (crop_size // 2, crop_size // 2), (0, 0)), 'constant')
        cells = np.pad(cells, ((crop_size // 2, crop_size // 2), (crop_size // 2, crop_size // 2)), 'constant')
    return image, cells, cells2labels


def _extend_slices_1d(slc, crop_size, max_size):
    """
    Extend a slice to be the size of crop size
    """
    d = crop_size - (slc.stop - slc.start)
    start = slc.start - (d // 2)
    stop = slc.stop + (d + 1) // 2
    if start < 0 or stop > max_size:
        raise Exception("Cell crop is out of bound of the image")
    return slice(start, stop)


def create_slices(slices, crop_size, bounds):
    """

    Args:
        slices: slices that bound the cell
        crop_size: the needed size of the crop
        bounds: shape of the image containing the cell

    Returns:
        new slices that bound the cell the size of crop_size
    """
    all_dim_slices = []
    for slc, cs, max_size in zip(slices, crop_size, bounds):
        all_dim_slices += [_extend_slices_1d(slc, cs, max_size)]
    return tuple(all_dim_slices)


def load_samples(images_dir, cells_dir, cells2labels_dir, images_names, crop_size, to_pad=False, channels=None):
    """

    Args:
        images_dir: path to the images
        cells_dir: path to the segmentation
        cells2labels_dir: path to mapping cells to labels
        images_names: names of images to load from the images_dir
        crop_size: the size of the crop of the cell
        channels: indices of channels to load from each image
    Returns:
        Array of CellCrop per cell in the dataset
    """
    images_dir = Path(images_dir)
    cells_dir = Path(cells_dir)
    cells2labels_dir = Path(cells2labels_dir)
    crops = []
    for image_id in images_names:
        image_path = glob.glob(str(images_dir / f"{image_id}.npz")) + \
                     glob.glob(str(images_dir / f"{image_id}.tiff"))
        cells_path = glob.glob(str(cells_dir / f"{image_id}.npz")) + \
                     glob.glob(str(cells_dir / f"{image_id}.tiff"))
        cells2labels_path = glob.glob(str(cells2labels_dir / f"{image_id}.npz")) + \
                            glob.glob(str(cells2labels_dir / f"{image_id}.txt"))
        image, cells, cl2lbl = load_image(image_path=image_path[0],
                                          cells_path=cells_path[0],
                                          cells2labels_path=cells2labels_path[0],
                                          channels=channels,
                                          to_pad=to_pad,
                                          crop_size=crop_size)

        objs = ndimage.find_objects(cells)
        for cell_id, obj in enumerate(objs, 1):
            try:
                slices = create_slices(obj, (crop_size, crop_size), cells.shape)
                label = cl2lbl[cell_id]
                crops.append(
                    CellCrop(cell_id=cell_id,
                             image_id=image_id,
                             label=label,
                             slices=slices,
                             cells=cells,
                             image=image))
            except Exception as e:
                pass
    return np.array(crops)


def load_crops(root_dir,
               channels_path,
               crop_size,
               train_set,
               val_set,
               to_pad=False,
               blacklist_channels=[]):
    """
    Given paths to the data, generate crops for all the cells in the data
    Args:
        root_dir:
        channels_path:
        crop_size: size of the environment to keep for each cell
        train_set: name of images to train on
        val_set: name of images to validate on
        to_pad: whether to pad the image with zeros in order to work on cell on the border
        blacklist_channels: channels to not use in the training/validation
    Returns:
        train_crops - list of crops from the train set
        val_crops - list of crops from the validation set
    """
    cell_types_dir = Path(root_dir) / "CellTypes"
    data_dir = cell_types_dir / "data" / "images"
    cells_dir = cell_types_dir / "cells"
    cells2labels_dir = cell_types_dir / "cells2labels"

    channels = read_channels(channels_path)
    channels_filtered = filter_channels(channels, blacklist_channels)
    channels = [i for i, c in channels_filtered]
    print(channels)
    print('Load training data...')
    train_crops = load_samples(images_dir=data_dir, cells_dir=cells_dir, cells2labels_dir=cells2labels_dir,
                               images_names=train_set, crop_size=crop_size, to_pad=to_pad,
                               channels=channels)

    print('Load validation data...')
    val_crops = load_samples(images_dir=data_dir, cells_dir=cells_dir, cells2labels_dir=cells2labels_dir,
                             images_names=val_set, crop_size=crop_size, to_pad=to_pad,
                             channels=channels)

    return train_crops, val_crops
