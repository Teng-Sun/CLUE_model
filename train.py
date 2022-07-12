import os
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict
from data_loader import get_loader
from model_carrier import Model_Carrier

import torch
import torch.nn as nn
from torch.nn import functional as F

def return_unk():
    return UNK

if __name__ == '__main__':

    random_name = str(random())
    random_seed = 123
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    train_config = get_config(mode = 'train')
    dev_config = get_config(mode = 'dev')
    test_iid_config = get_config(mode = 'test_iid')
    test_ood_config = get_config(mode = 'test_ood')

    print(train_config)

    train_loader, train_length = get_loader(train_config, shuffle = True)
    train_config.train_samples = train_length
    data_loader = {
        'train': train_loader,
        'dev': get_loader(dev_config, shuffle = False)[0],
        'test_iid': get_loader(test_iid_config, shuffle = False)[0],
        'test_ood': get_loader(test_ood_config, shuffle = False)[0],
    }

    model_carrier = Model_Carrier(train_config, data_loader, is_train = True)

    model_carrier.build()
    model_carrier.train()