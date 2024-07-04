import numpy as np
import os


def safe_save(save_name: str, array: np.ndarray) -> None:
    """
    If a folder is specified and does not yet exist, it will be created automatically.
    :param save_name: full path + data name
    :param array: array to save
    :return:
    """
    # create folder if not exists
    folder, data_name = os.path.split(save_name)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if data_name[-3:] == 'npy':
        np.save(save_name, array)
    else:
        np.save(save_name + '.npy', array)


def cumulative_sum(a: np.ndarray, n: int, axis: int | None = None) -> np.ndarray:
    """
    Sums an array a over a given interval n. Mathematically equivalent to a convolution with a rectangular function
    of width n.
    :param a: array
    :param n: interval width
    :param axis:
    :return:
    """
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]


def get_element_by_interval(l, a, b):
    if not l or b <= 0:
        return None

    index = int((a - 1) // b)
    return l[index % len(l)] if l else None
