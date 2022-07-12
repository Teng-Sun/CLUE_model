import torch
import numpy as np


def to_gpu(x, gpu_id = None):
    """ Tensor => Variable """
    if torch.cuda.is_available() and gpu_id is not None:
        x = x.cuda(gpu_id)
    return x


def to_cpu(x):
    """ Variable => Tensor """
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


def change_to_classify(y, config):
    """ 2 classes or 7 classes """
    if config.output_size == 2:
        return y >= 0

    if config.output_size == 7:
        y = np.clip(y, a_min = -3., a_max = 3.)
        return np.round(y) + 3