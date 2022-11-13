"""
From: https://github.com/thuiar/Self-MM
Paper: Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis
"""
# self supervised multimodal multi-task learning network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from utils import to_gpu

from models.subNets.BertTextEncoder import BertTextEncoder

class SELF_MM(nn.Module):
    def __init__(self, config):
        super(SELF_MM, self).__init__()
        # text subnets
        self.config = config
        self.text_model = BertTextEncoder(use_finetune = True, transformers = 'bert', pretrained = "bert-base-uncased")
        
        v_lstm_hidden_size = video_out = post_video_dim = config.selfmm_video_hidden_size
        a_lstm_hidden_size = audio_out = post_audio_dim = config.selfmm_audio_hidden_size
        post_fusion_dim = config.selfmm_fusion_hidden_size
        post_text_dim = config.selfmm_text_hidden_size
        
        post_text_dropout = 0.1
        post_audio_dropout = 0.1
        post_video_dropout = 0
        post_fusion_dropout = 0

        # audio-vision subnets
        text_out = config.bert_text_size
        audio_in = config.acoustic_size
        video_in = config.visual_size

        self.audio_model = AuViSubNet(audio_in, a_lstm_hidden_size, audio_out, num_layers = 1, dropout = 0)
        self.video_model = AuViSubNet(video_in, v_lstm_hidden_size, video_out, num_layers = 1, dropout = 0)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p = post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(text_out + video_out + audio_out, post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(post_fusion_dim, post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(post_fusion_dim, 1)

        self.post_fusion_layer2_2 = nn.Linear(post_fusion_dim, post_fusion_dim)
        self.post_fusion_layer2_3 = nn.Linear(post_fusion_dim, config.output_size)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p = post_text_dropout)
        self.post_text_layer_1 = nn.Linear(text_out, post_text_dim)
        self.post_text_layer_2 = nn.Linear(post_text_dim, post_text_dim)
        self.post_text_layer_3 = nn.Linear(post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p = post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(audio_out, post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(post_audio_dim, post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p = post_video_dropout)
        self.post_video_layer_1 = nn.Linear(video_out, post_video_dim)
        self.post_video_layer_2 = nn.Linear(post_video_dim, post_video_dim)
        self.post_video_layer_3 = nn.Linear(post_video_dim, 1)
        
        self.tasks = "MTAV"
        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }
        self.feature_map = {
            'fusion': to_gpu(torch.zeros(config.train_samples, post_fusion_dim, requires_grad=False), gpu_id = config.gpu_id),
            'text': to_gpu(torch.zeros(config.train_samples, post_text_dim, requires_grad=False), gpu_id = config.gpu_id),
            'audio': to_gpu(torch.zeros(config.train_samples, post_audio_dim, requires_grad=False), gpu_id = config.gpu_id),
            'vision': to_gpu(torch.zeros(config.train_samples, post_video_dim, requires_grad=False), gpu_id = config.gpu_id),
        }
        self.label_map = {
            'fusion': to_gpu(torch.zeros(config.train_samples, requires_grad=False), gpu_id = config.gpu_id),
            'text': to_gpu(torch.zeros(config.train_samples, requires_grad=False), gpu_id = config.gpu_id),
            'audio': to_gpu(torch.zeros(config.train_samples, requires_grad=False), gpu_id = config.gpu_id),
            'vision': to_gpu(torch.zeros(config.train_samples, requires_grad=False), gpu_id = config.gpu_id),
        }
        self.center_map = {
            'fusion': {
                'pos': to_gpu(torch.zeros(post_fusion_dim, requires_grad=False), gpu_id = config.gpu_id),
                'neg': to_gpu(torch.zeros(post_fusion_dim, requires_grad=False), gpu_id = config.gpu_id),
            },
            'text': {
                'pos': to_gpu(torch.zeros(post_text_dim, requires_grad=False), gpu_id = config.gpu_id),
                'neg': to_gpu(torch.zeros(post_text_dim, requires_grad=False), gpu_id = config.gpu_id),
            },
            'audio': {
                'pos': to_gpu(torch.zeros(post_audio_dim, requires_grad=False), gpu_id = config.gpu_id),
                'neg': to_gpu(torch.zeros(post_audio_dim, requires_grad=False), gpu_id = config.gpu_id),
            },
            'vision': {
                'pos': to_gpu(torch.zeros(post_video_dim, requires_grad=False), gpu_id = config.gpu_id),
                'neg': to_gpu(torch.zeros(post_video_dim, requires_grad=False), gpu_id = config.gpu_id),
            }
        }
        self.dim_map = {
            'fusion': to_gpu(torch.tensor(post_fusion_dim), gpu_id = config.gpu_id).float(),
            'text': to_gpu(torch.tensor(post_text_dim), gpu_id = config.gpu_id).float(),
            'audio': to_gpu(torch.tensor(post_audio_dim), gpu_id = config.gpu_id).float(),
            'vision': to_gpu(torch.tensor(post_video_dim), gpu_id = config.gpu_id).float(),
        }

    def forward(self, batch_sample, epoch_info):
        audio = batch_sample['audio']
        video = batch_sample['visual']
        lengths = batch_sample['lengths']
        text = torch.cat((
            batch_sample['bert_sentences'].unsqueeze(1),
            batch_sample['bert_sentence_att_mask'].unsqueeze(1),
            batch_sample['bert_sentence_types'].unsqueeze(1)
        ), dim = 1)
        
        text = self.text_model(text)[:,0,:]

        audio = self.audio_model(audio, lengths)
        video = self.video_model(video, lengths)
        
        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        x_f2 = F.relu(self.post_fusion_layer2_2(fusion_h), inplace=False)
        output_fusion2 = self.post_fusion_layer2_3(x_f2)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        outputs = {
            'M': output_fusion,
            'T': output_text,
            'A': output_audio,
            'V': output_video,
        }

        base_loss = 0
        indexes = to_gpu(batch_sample['index'].view(-1))

        for key in self.tasks:
            base_loss += self.weighted_loss(
                outputs[key],
                self.label_map[self.name_map[key]][indexes],
                indexes = indexes,
                mode = self.name_map[key]
            )
        
        f_fusion = fusion_h.detach()
        f_text = text_h.detach()
        f_audio = audio_h.detach()
        f_vision = video_h.detach()
        
        res = {
            'o_mutimodel': output_fusion2,
            'o_text': output_text,
            'loss': base_loss,
        }

        if epoch_info['epoch_index'] < 0: return res
        if epoch_info['epoch_index'] > 1:
            self.update_labels(f_fusion, f_text, f_audio, f_vision, epoch_info['epoch_index'], indexes, outputs)

        self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
        self.update_centers()

        return res
    
    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss
    
    def update_features(self, f_fusion, f_text, f_audio, f_vision, indexes):
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text
        self.feature_map['audio'][indexes] = f_audio
        self.feature_map['vision'][indexes] = f_vision

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.config.selfmm_excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
        update_single_center(mode='text')
        update_single_center(mode='audio')
        update_single_center(mode='vision')
    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels
    
    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8
        def update_single_label(f_single, mode):
            d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1) 
            d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1) 
            delta_s = (d_sn - d_sp) / (d_sp + MIN)
            # d_s_pn = torch.norm(self.center_map[mode]['pos'] - self.center_map[mode]['neg'], dim=-1)
            # delta_s = (d_sn - d_sp) / (d_s_pn + MIN)
            alpha = delta_s / (delta_f + MIN)

            new_labels = 0.5 * alpha * self.label_map['fusion'][indexes] + \
                        0.5 * (self.label_map['fusion'][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.config.selfmm_H, max=self.config.selfmm_H)
            # new_labels = torch.tanh(new_labels) * self.config.H

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        d_fp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1) 
        d_fn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1) 
        # d_f_pn = torch.norm(self.center_map['fusion']['pos'] - self.center_map['fusion']['neg'], dim=-1)
        # delta_f = (d_fn - d_fp) / (d_f_pn + MIN)
        delta_f = (d_fn - d_fp) / (d_fp + MIN)
        
        update_single_label(f_text, mode='text')
        update_single_label(f_audio, mode='audio')
        update_single_label(f_vision, mode='vision')
    
class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        config:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.cpu()
        packed_sequence = pack_padded_sequence(x, lengths)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1