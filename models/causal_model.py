import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.misa_model import MISA
from models.selfmm_model import SELF_MM
from models.magbert_model import BERT_MAG
import torch.nn.functional as F

from models.subNets.BertTextEncoder import BertTextEncoder

class Causal_Model(nn.Module):
    def __init__(self, config):
        super(Causal_Model, self).__init__()

        self.config = config
        self.fusion_mode = config.fusion_mode
        self.output_size = config.output_size
        self.hidden_size = config.tmodel_hidden_size
        self.embedding_size = config.tmodel_embedding_size

        self.size = 7
        # loss function
        if self.size != 7:
            self.classify_criterion = nn.CrossEntropyLoss(reduction = "mean")
        else:
            # self.classify_criterion = nn.L1Loss(reduction = "mean")
            self.classify_criterion = nn.KLDivLoss(reduction = "batchmean")

        # multimodel -- basemodel
        if self.config.base_model == "misa_model":
            self.base_model = MISA(config)
        elif self.config.base_model == "selfmm_model":
            self.base_model = SELF_MM(config)
        elif self.config.base_model == "magbert_model":
            self.base_model = BERT_MAG(config)
        else:
            raise NameError('No {} model can be found'.format(self.config.base_model))
        

        # multimodel -- textmodel
        rnn = nn.LSTM if self.config.tmodel_rnncell == "lstm" else nn.GRU
        self.trnn1 = rnn(self.embedding_size, self.embedding_size, bidirectional=True)
        self.trnn2 = rnn(2 * self.embedding_size, self.embedding_size, bidirectional=True)
        self.tlayer_norm = nn.LayerNorm((2 * self.embedding_size,))

        self.text_mlp = nn.Sequential(
            nn.Linear(self.embedding_size * 4, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.output_size)
        )

        self.constant = nn.Parameter(torch.tensor(1.0))
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, batch_sample, labels, epoch_info = {'batch_index': 0, 'epoch_index': -1}):
        sentences = batch_sample['text']
        ban_sentences = batch_sample['ban_text']
        lengths = batch_sample['lengths']
        batch_size = lengths.size(0)

        # basemodel output
        base_output = self.base_model(batch_sample, epoch_info)
        o_mutimodel = base_output['o_mutimodel']
        base_loss = base_output['loss']

        # textmodel output
        final_h1t, final_h2t = self.extract_features(ban_sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        emb_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        o_text = self.text_mlp(emb_text)

        o_mutimodel_c = self.constant * torch.ones_like(o_mutimodel).cuda(self.config.gpu_id)
        o1_fusion = self.fusion_fuction(o_mutimodel, o_text)
        o2_fusion = self.fusion_fuction(o_mutimodel_c, o_text)
        
        output = {
            'o_text': o_text,
            'o_mutimodel': o_mutimodel,
            'o1_fusion': o1_fusion,
            'o2_fusion': o2_fusion
        }

        if labels is None:
            output['loss'] = 0
            return output

        if self.size != 7:
            labels = labels.long()
        else:
            labels = labels.float()
            log_softmax = torch.nn.LogSoftmax(dim = 1)

        if self.config.only_base_model:
            o1_classify_lose = self.classify_criterion(o_mutimodel, labels)
            output['loss'] = base_loss + o1_classify_lose
            output['o1_classify_lose'] = o1_classify_lose
        elif self.config.only_text_model:
            o2_classify_lose = self.classify_criterion(o_text, labels)
            output['loss'] = output['o1_classify_lose'] = o2_classify_lose
        else:
            kl_loss = 0
            
            if self.size != 7:
                o1_classify_lose = self.classify_criterion(o1_fusion, labels)
                o2_classify_lose = self.classify_criterion(o_text, labels)
            else:
                o1_classify_lose = self.classify_criterion(log_softmax(o1_fusion), labels)
                o2_classify_lose = self.classify_criterion(log_softmax(o_text), labels)
            
            if self.config.klloss: kl_loss = self.kl_loss(o1_fusion, self.fusion_fuction(o_mutimodel_c, o_text.detach()))
            output['loss'] = self.config.o1_weight * (base_loss + o1_classify_lose) + self.config.o2_weight * o2_classify_lose + kl_loss
            output['o1_classify_lose'] = o1_classify_lose

        return output

    def fusion_fuction(self, o1, o2):
        eps = 1e-12
        if self.fusion_mode == "sum":
            o_fusion = torch.log(torch.sigmoid(o1 + o2) + eps)

        if self.fusion_mode == "hm":
            o = torch.sigmoid(o1) * torch.sigmoid(o2)
            o_fusion = torch.log(o + eps) - torch.log1p(o)
        
        if self.fusion_mode == 'rubi':
            o_fusion = o1 * torch.sigmoid(o2)

        return o_fusion
    
    
    def kl_loss(self, o1_fusion, o2_fusion):
        p_te = nn.functional.softmax(o1_fusion, -1).clone().detach()
        p_nde = nn.functional.softmax(o2_fusion, -1)
        kl_loss = -p_te * p_nde.log()
        kl_loss = kl_loss.sum(1).mean()

        return kl_loss
    

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        lengths = lengths.cpu()
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.tmodel_rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.tmodel_rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2