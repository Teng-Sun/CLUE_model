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

from create_dataset import Data_Reader, PAD

def return_unk():
    return UNK

def change(i):    
    if i < -3: return 0
    elif i > 3: return 6
    else: return int(round(i)) + 3

def get_ban_list(data, train_config):
    ban_number = 0
    all_number = 0
    word_dict = {}
    
    
    for sample in data:
        label = sample[1][0][0]
        word_list = sample[0][3]
        all_number += len(word_list)
        
        for word in word_list:
            if train_config.output_size == 7:
                if word not in word_dict:
                    word_dict[word] = [0, 0, 0, 0, 0, 0, 0]
                new_label = change(label)
                word_dict[word][new_label] += 1

            if train_config.output_size == 2:
                if word not in word_dict:
                    word_dict[word] = [0, 0]
                if label > 0: word_dict[word][0] += 1
                elif label < 0: word_dict[word][1] += 1

    new_word_dict = {}
    
    if train_config.output_size == 2:
        for word, count_list in word_dict.items():
            if word not in new_word_dict:
                total = count_list[0] + count_list[1]
                if total <= train_config.min_number: continue
                new_word_dict[word] = abs(count_list[0] / total - count_list[1] / total)
    
    if train_config.output_size == 7:
        for word, count_list in word_dict.items():
            if word not in new_word_dict:
                total = 0
                for count in count_list:
                    total += count
                if total <= train_config.min_number: continue
                ban_number += total 
                new_word_dict[word] = 0
                for count in count_list:
                    new_word_dict[word] += abs(count / total - 1.0 / 7.0)

    word_dict = sorted(new_word_dict.items(), key = lambda x: x[1], reverse = True)
    
    ban_word_list = []
    for word, rate in word_dict[int(len(word_dict) * train_config.split_rate): ]:
        ban_word_list.append(word)
    
    ban_word_dict = {}
    for ban_word in ban_word_list:
        ban_word_dict[ban_word] = 0
    
    print(all_number, ban_number)
    return ban_word_dict

random_seed_list = [223, 123, 323]

if __name__ == '__main__':
    avg_best_metrics = {
        'over_metrics': {"acc7": 0, "f_score_nonzero": 0, "f_score": 0, "acc2_nonzero": 0, "acc2": 0},
        'o1_metrics': {"acc7": 0, "f_score_nonzero": 0, "f_score": 0, "acc2_nonzero": 0, "acc2": 0},
        'o2_metrics': {"acc7": 0, "f_score_nonzero": 0, "f_score": 0, "acc2_nonzero": 0, "acc2": 0}
    }
    
    train_config = get_config(mode = 'train')
    dev_config = get_config(mode = 'dev')
    test_iid_config = get_config(mode = 'test_iid')
    test_ood_config = get_config(mode = 'test_ood')

    dataset = Data_Reader(train_config)
    data, word2id, pretrained_emb = dataset.get_data(train_config.mode)

    # 获取抹除词表
    ban_word_list = get_ban_list(data, train_config)

    acc2_nonzero_list = []
    acc2_list = []
    acc7_list = []

    print(train_config)

    for random_seed in random_seed_list:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)

        train_loader, train_length = get_loader(train_config, shuffle = True, ban_word_list = ban_word_list)
        train_config.train_samples = train_length
        
        data_loader = {
            'train': train_loader,
            'dev': get_loader(dev_config, shuffle = False)[0],
            'test_iid': get_loader(test_iid_config, shuffle = False)[0],
            'test_ood': get_loader(test_ood_config, shuffle = False)[0],
        }

        model_carrier = Model_Carrier(train_config, data_loader, is_train = True)

        model_carrier.build()
        best_metrics = model_carrier.train()

        for key, value in best_metrics['o1']['ood_metrics']['over_metrics'].items():
            avg_best_metrics['over_metrics'][key] += value

        for key, value in best_metrics['o1']['ood_metrics']['o1_metrics'].items():
            avg_best_metrics['o1_metrics'][key] += value
        
        for key, value in best_metrics['o1']['ood_metrics']['o2_metrics'].items():
            avg_best_metrics['o2_metrics'][key] += value
        
        acc2_list.append(best_metrics['o1']['ood_metrics']['over_metrics']['acc2'])
        acc2_nonzero_list.append((random_seed, best_metrics['o1']['ood_metrics']['over_metrics']['acc2_nonzero']))
        acc7_list.append(best_metrics['o1']['ood_metrics']['over_metrics']['acc7'])
        
    
    for key1, value1 in avg_best_metrics.items():
        for key2, value2 in value1.items():
            avg_best_metrics[key1][key2] = value2 / 3.0
    
    print("acc2: ")
    print((acc2_list[0] + acc2_list[1] + acc2_list[2]) / 3.0)
    print("acc2_nonzero: ")
    print((acc2_nonzero_list[0][1] + acc2_nonzero_list[1][1] + acc2_nonzero_list[2][1]) / 3.0)
    print("acc7: ")
    print((acc7_list[0] + acc7_list[1] + acc7_list[2]) / 3.0)
    
    print(acc7_list)
    print(acc2_list)
    print(acc2_nonzero_list)
