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


def cumulative_sum(a: np.ndarray, n: int) -> np.ndarray:
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]