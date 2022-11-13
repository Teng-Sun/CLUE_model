import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

def return_unk():
    return UNK

class Data_Reader:
    def __init__(self, args):
        DATA_PATH = '{}/{}/{}/'.format(args.datapath, args.data.upper(), args.dataset_name)
        
        self.train = []
        self.dev = []
        self.test_iid = []
        self.test_ood = []
        self.pretrained_emb, self.word2id = torch.load(DATA_PATH + 'embedding_and_mapping.pt')

        train_data = load_pickle(DATA_PATH + 'train.pkl')
        for index, i in enumerate(train_data):
            (content_data, label, segment) = i
            label = np.array([[label[0][0]]])
            self.train.append((content_data, label, segment))

        dev_data = load_pickle(DATA_PATH + 'dev.pkl')
        for index, i in enumerate(dev_data):
            (content_data, label, segment) = i
            label = np.array([[label[0][0]]])
            self.dev.append((content_data, label, segment))

        test_iid_data = load_pickle(DATA_PATH + 'test_IID.pkl')
        for index, i in enumerate(test_iid_data):
            (content_data, label, segment) = i
            label = np.array([[label[0][0]]])
            self.test_iid.append((content_data, label, segment))

        test_ood_data = load_pickle(DATA_PATH + 'test_OOD.pkl')
        for index, i in enumerate(test_ood_data):
            (content_data, label, segment) = i
            label = np.array([[label[0][0]]])
            self.test_ood.append((content_data, label, segment))
        
    def get_data(self, mode):
        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test_iid":
            return self.test_iid, self.word2id, self.pretrained_emb
        elif mode == "test_ood":
            return self.test_ood, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()