import os
import math
from math import isnan
import re
import json
import pickle
import gensim
# from gevent import config
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from models.causal_model import Causal_Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import to_gpu

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

class Model_Carrier(object):
    def __init__(self, config, data_loader, is_train = True, model = None):

        self.config = config
        self.epoch_i = 0
        self.train_data_loader = data_loader['train']
        self.dev_data_loader = data_loader['dev']
        self.test_iid_data_loader = data_loader['test_iid']
        self.test_ood_data_loader = data_loader['test_ood']
        self.is_train = is_train
        self.model = model

        self.best_metrics = {
            'epoch': 0,
            'alpha': 0,
            'over': None,
            'o1': None,
        }
        self.best_valid_acc2_loss = 0
        self.best_valid_acc7_loss = 0

    # print(self.best_valid_acc2_loss)
    
    def build(self, cuda=True):
        if self.model is None:
            self.model = Causal_Model(self.config)
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda(self.config.gpu_id)
        
        for name, param in self.model.named_parameters():
            if self.config.use_bert_frozen:
                if "bertmodel.encoder.layer" in name or "text_model.model.encoder" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)

        if self.config.base_model == 'misa_model':
            model_params = list(self.model.named_parameters())
            text_model_params = [p for n, p in model_params if 'base_model' not in n]
            base_model_params = [p for n, p in model_params if 'base_model' in n]
            optimizer_parameters = [
                {'params': text_model_params, 'weight_decay': self.config.tmodel_weight_decay, 'lr': self.config.tmodel_learning_rate},
                {'params': base_model_params, 'weight_decay': self.config.misa_weight_decay, 'lr': self.config.misa_learning_rate},
            ]
            self.optimizer = self.config.optimizer(optimizer_parameters)
        
        if self.config.base_model == 'selfmm_model':
            model_params = list(self.model.named_parameters())
            text_model_params = [p for n, p in model_params if 'base_model' not in n]
            base_model_params = [p for n, p in model_params if 'base_model' in n]
            optimizer_parameters = [
                {'params': text_model_params, 'weight_decay': self.config.tmodel_weight_decay, 'lr': self.config.tmodel_learning_rate},
                {'params': base_model_params, 'weight_decay': self.config.selfmm_weight_decay, 'lr': self.config.selfmm_learning_rate},
            ]
            self.optimizer = self.config.optimizer(optimizer_parameters)
        
        if self.config.base_model == 'magbert_model':
            model_params = list(self.model.named_parameters())
            text_model_params = [p for n, p in model_params if 'base_model' not in n]
            base_model_params = [p for n, p in model_params if 'base_model' in n]
            optimizer_parameters = [
                {'params': text_model_params, 'weight_decay': self.config.tmodel_weight_decay, 'lr': self.config.tmodel_learning_rate},
                {'params': base_model_params, 'weight_decay': self.config.magbert_weight_decay, 'lr': self.config.magbert_learning_rate},
            ]
            self.optimizer = self.config.optimizer(optimizer_parameters)
        
    def pretrain(self):
        if self.config.base_model == 'selfmm_model':
            with tqdm(self.train_data_loader) as td:
                for batch_data in td:
                    labels_m = to_gpu(batch_data['labels'].view(-1), gpu_id = self.config.gpu_id)
                    indexes = batch_data['index'].view(-1)
                    self.model.base_model.init_labels(indexes, labels_m)

    def overtrain(self, epoch):
        best_ood_acc2_avg = 0
        best_ood_acc7 = 0
        best_alpha = 0
        best_ood_metrics = None

        valid_metrics = self.eval(mode = "dev")
        iid_metrics = self.eval(mode = "iid_test")
        print(valid_metrics['o1_loss'])

        if self.config.output_size == 2:  
            for alpha in self.config.alpha_list:
                ood_metrics = self.eval(mode = "ood_test", alpha = alpha)
                print(ood_metrics['over_metrics']['acc2'], ood_metrics['over_metrics']['acc2_nonzero'])
                ood_acc2_avg = (ood_metrics['over_metrics']['acc2'] + ood_metrics['over_metrics']['acc2_nonzero']) / 2.0
                if ood_acc2_avg > best_ood_acc2_avg:
                    best_ood_acc2_avg = ood_acc2_avg
                    best_ood_metrics = ood_metrics
                    best_alpha = alpha
            
            
            print(valid_metrics)
            print(best_ood_metrics)
            '''
            for alpha in self.config.alpha_list:
                ood_metrics = self.eval(mode = "iid_test", alpha = alpha)
                print(ood_metrics['over_metrics']['acc2'], ood_metrics['over_metrics']['acc2_nonzero'])
                ood_acc2_avg = (ood_metrics['over_metrics']['acc2'] + ood_metrics['over_metrics']['acc2_nonzero']) / 2.0
                if ood_acc2_avg > best_ood_acc2_avg:
                    best_ood_acc2_avg = ood_acc2_avg
                    best_ood_metrics = ood_metrics
                    best_alpha = alpha

            print(best_ood_metrics)
            '''
            if (valid_metrics['o1_metrics']['acc2_nonzero'] + valid_metrics['o1_metrics']['acc2']) / 2 > self.best_valid_acc2_loss:
                self.best_valid_acc2_loss = (valid_metrics['o1_metrics']['acc2_nonzero'] + valid_metrics['o1_metrics']['acc2']) / 2
                self.best_metrics['o1'] = {
                    'valid_metrics': valid_metrics,
                    'iid_metrics': iid_metrics,
                    'ood_metrics': best_ood_metrics,
                }
                self.best_metrics['epoch'] = epoch
                self.best_metrics['alpha'] = best_alpha
                state_dict = {'net': self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch': epoch}
                torch.save(state_dict, '{}/{}_{}_{}_bestmodel.pth'.format(
                    self.config.model_savepath,
                    self.config.model_index,
                    self.config.base_model,
                    self.config.dataset_name,
                ))

        if self.config.output_size == 7:
            for alpha in self.config.alpha_list:
                ood_metrics = self.eval(mode = "ood_test", alpha = alpha)
                print(ood_metrics['over_metrics']['acc7'])
                if ood_metrics['over_metrics']['acc7'] > best_ood_acc7:
                    best_ood_acc7 = ood_metrics['over_metrics']['acc7']
                    best_ood_metrics = ood_metrics
                    best_alpha = alpha

            print(valid_metrics)
            print(best_ood_metrics)

            if valid_metrics['o1_metrics']['acc7'] >= self.best_valid_acc7_loss:
                self.best_valid_acc7_loss = valid_metrics['o1_metrics']['acc7']
                self.best_metrics['o1'] = {
                    'valid_metrics': valid_metrics,
                    'iid_metrics': iid_metrics,
                    'ood_metrics': best_ood_metrics,
                }
                self.best_metrics['epoch'] = epoch
                self.best_metrics['alpha'] = best_alpha
                state_dict = {'net': self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch': epoch}
                torch.save(state_dict, '{}/{}_{}_{}_bestmodel.pth'.format(
                    self.config.model_savepath,
                    self.config.model_index,
                    self.config.base_model,
                    self.config.dataset_name,
                ))

        print(valid_metrics['o1_loss'])

    def train(self):
        self.pretrain()

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 20, eta_min = 1e-5)
        self.eval(mode = "dev")

        for epoch in range(self.config.n_epoch):
            self.model.train()

            train_loss = []
            for index, batch_sample in enumerate(self.train_data_loader):
                self.model.zero_grad()

                output = self.model(
                    batch_sample,
                    batch_sample['labels_classify'],
                    {'batch_index': index, 'epoch_index': epoch + 1}
                )

                loss = output['loss']
                loss.backward()
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.config.clip)
                self.optimizer.step()
                train_loss.append(loss.item())
            
            self.overtrain(epoch)
            lr_scheduler.step()
            print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

        print('over_train!')
        print(json.dumps(self.best_metrics, indent = 4, ensure_ascii = False))
        return self.best_metrics
    
    def eval(self, mode = None, alpha = 1):
        self.model.eval()
        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "iid_test":
            dataloader = self.test_iid_data_loader
        elif mode == "ood_test":
            dataloader = self.test_ood_data_loader
        elif mode == "train":
            dataloader = self.train_data_loader

        y_true = []
        y_over_fusion = []
        y_o1_fusion = []
        y_o2_fusion = []

        with torch.no_grad():
            o1_loss = []
            for index, batch_sample in enumerate(dataloader):
                labels = batch_sample['labels']
                output = self.model(batch_sample, batch_sample['labels_classify'], {'batch_index': index, 'epoch_index': -1})

                o1_fusion = output['o1_fusion']
                o2_fusion = output['o2_fusion']

                o1_loss.append(np.mean(output['o1_classify_lose'].detach().cpu().numpy()))

                if self.config.only_base_model:
                    o1_fusion = output['o_mutimodel']
                    over_fusion = o1_fusion
                    o2_fusion = o1_fusion
                elif self.config.only_text_model:
                    o1_fusion = output['o_text']
                    over_fusion = o1_fusion
                    o2_fusion = o1_fusion
                else:
                    over_fusion = o1_fusion - alpha * o2_fusion
                
                _, pred_over_fusion = torch.max(over_fusion, 1)
                _, pred_o1_fusion = torch.max(o1_fusion, 1)
                _, pred_o2_fusion = torch.max(o2_fusion, 1)
                
                y_true.append(labels.detach().cpu().numpy())
                y_o1_fusion.append(pred_o1_fusion.detach().cpu().numpy())
                y_o2_fusion.append(pred_o2_fusion.detach().cpu().numpy())
                y_over_fusion.append(pred_over_fusion.detach().cpu().numpy())

            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_o1_fusion = np.concatenate(y_o1_fusion, axis=0).squeeze()
            y_o2_fusion = np.concatenate(y_o2_fusion, axis=0).squeeze()
            y_over_fusion = np.concatenate(y_over_fusion, axis=0).squeeze()

            over_metrics = self.calc_metrics(y_true, y_over_fusion, mode, False)
            o1_metrics = self.calc_metrics(y_true, y_o1_fusion, mode, False)
            o2_metrics = self.calc_metrics(y_true, y_o2_fusion, mode, False)

        return {
            'over_metrics': over_metrics,
            'o1_metrics': o1_metrics,
            'o2_metrics': o2_metrics,
            'o1_loss': float(np.mean(o1_loss)),
        }

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) - 3 == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode = None, to_print = False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """
        test_preds = y_pred
        test_truth = y_true

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        acc7 = self.multiclass_acc(y_pred, test_truth_a7)

        f_score_nonzero = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
        f_score = f1_score((test_preds > 0), (test_truth >= 0), average='weighted')

        # pos - neg
        if self.config.output_size == 7:
            binary_truth = (test_truth[non_zeros] >= 0)
            binary_preds = (test_preds[non_zeros] >= 3)
        if self.config.output_size == 2 or self.config.output_size == 3:
            binary_truth = (test_truth[non_zeros] >= 0)
            binary_preds = (test_preds[non_zeros] > 0)

        acc2_nonzero = accuracy_score(binary_truth, binary_preds)

        # non-neg - neg
        if self.config.output_size == 7:
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 3)
        if self.config.output_size == 2:
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds > 0)
        acc2 = accuracy_score(binary_truth, binary_preds)

        metrics_output = {
            'acc7': acc7,
            'f_score_nonzero': f_score_nonzero,
            'f_score': f_score,
            'acc2_nonzero': acc2_nonzero,
            'acc2': acc2
        }

        return metrics_output