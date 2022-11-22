import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
import math

import sys
sys.path.append("..")
from utils import to_gpu, DiffLoss, MSE, SIMSE, CMD, ReverseLayerF

def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)

# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.misa_embedding_size
        self.bert_text_size = config.bert_text_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size
        self.hidden_size = config.misa_hidden_size
    
        self.output_size = config.num_classes
        self.dropout_rate = config.misa_dropout
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction = "mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction = "mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()

        rnn = nn.LSTM if self.config.misa_rnncell == "lstm" else nn.GRU

        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        
        self.vrnn1 = rnn(self.visual_size, self.visual_size, bidirectional=True)
        self.vrnn2 = rnn(2*self.visual_size, self.visual_size, bidirectional=True)
        
        self.arnn1 = rnn(self.acoustic_size, self.acoustic_size, bidirectional=True)
        self.arnn2 = rnn(2*self.acoustic_size, self.acoustic_size, bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.misa_use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=self.bert_text_size, out_features=self.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=self.text_size*4, out_features=self.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=self.visual_size*4, out_features=self.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(self.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=self.acoustic_size*4, out_features=self.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(self.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        self.private_t2 = nn.Sequential()
        self.private_t2.add_module('private_t2_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_t2.add_module('private_t_activation2_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        
        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.misa_use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(self.dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=self.hidden_size, out_features=3))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################
        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=self.hidden_size, out_features=4))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.hidden_size*6, out_features=self.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.hidden_size*3, out_features= self.output_size))

        self.vlayer_norm = nn.LayerNorm((self.visual_size*2,))
        self.alayer_norm = nn.LayerNorm((self.acoustic_size*2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        lengths = lengths.cpu()
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.misa_rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.misa_rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, batch_sample):
        visual = batch_sample['visual']
        acoustic = batch_sample['audio']
        lengths = batch_sample['lengths']
        bert_sent = batch_sample['bert_sentences']
        bert_sent_type = batch_sample['bert_sentence_types']
        bert_sent_mask = batch_sample['bert_sentence_att_mask']
        batch_size = lengths.size(0)
        
        bert_output = self.bertmodel(input_ids = bert_sent, attention_mask = bert_sent_mask, token_type_ids = bert_sent_type)      
        bert_output = bert_output[0]
        # masked mean
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim = 1, keepdim = True)  
        bert_output = torch.sum(masked_output, dim = 1, keepdim = False) / mask_len
        utterance_text = bert_output
        
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim = 2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim = 2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        if not self.config.misa_use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.misa_reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.misa_reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.misa_reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        om = self.fusion(h)

        diff_loss = self.get_diff_loss()
        domain_loss = self.get_domain_loss()
        recon_loss = self.get_recon_loss()
        cmd_loss = self.get_cmd_loss()

        if self.config.misa_use_cmd_sim:
            similarity_loss = cmd_loss
        else:
            similarity_loss = domain_loss

        loss = self.config.misa_diff_weight * diff_loss + self.config.misa_sim_weight * similarity_loss + self.config.misa_recon_weight * recon_loss
        
        output = {
            'o_mutimodel': om,
            'loss': loss,
        }

        return output
    
    def forward(self, batch_sample, epoch_info):
        o = self.alignment(batch_sample)
        return o
    
    def reconstruct(self,):
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)
    
    def get_recon_loss(self):
        loss = self.loss_recon(self.utt_t_recon, self.utt_t_orig)
        loss += self.loss_recon(self.utt_v_recon, self.utt_v_orig)
        loss += self.loss_recon(self.utt_a_recon, self.utt_a_orig)
        loss = loss / 3.0
        return loss

    def get_domain_loss(self):
        if self.config.misa_use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.domain_label_t
        domain_pred_v = self.domain_label_v
        domain_pred_a = self.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
        domain_true_v = to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)
    
    def get_cmd_loss(self):
        if not self.config.misa_use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.utt_shared_t, self.utt_shared_v, 5)
        loss += self.loss_cmd(self.utt_shared_t, self.utt_shared_a, 5)
        loss += self.loss_cmd(self.utt_shared_a, self.utt_shared_v, 5)
        loss = loss/3.0

        return loss
    
    def get_diff_loss(self):
        shared_t = self.utt_shared_t
        shared_v = self.utt_shared_v
        shared_a = self.utt_shared_a
        private_t = self.utt_private_t
        private_v = self.utt_private_v
        private_a = self.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss