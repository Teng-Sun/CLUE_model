import torch
import numpy as np
import math

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
    if config.soft_label:
        over_sample = []
        over_sample2 = []
        for i in y:
            sig = math.sqrt(config.variance)  # 标准差δ
            x = np.array([-3, -2, -1, 0, 1, 2, 3])
            y_sig = np.exp(-(x - i[0].item()) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
            data = y_sig / np.sum(y_sig)
            over_sample.append(data)
            over_sample2.append([data[0] + data[1] + data[2], data[3] + data[4] + data[5] + data[6]])
        if config.output_size == 7:
            return torch.tensor(over_sample)
        else:
            return torch.tensor(over_sample2)
    else:
        if config.output_size == 2:
            return y >= 0
        
        if config.output_size == 7:
            y = np.clip(y, a_min = -3., a_max = 3.)
            return np.round(y) + 3
