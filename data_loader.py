import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import Data_Reader, PAD
from utils import to_gpu, change_to_classify

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MSADataset(Dataset):
    def __init__(self, config):

        dataset = Data_Reader(config)
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]
        config.bert_text_size = 768
        
        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

    def __getitem__(self, index):
        return {
            "data": self.data[index],
            "index": index
        }

    def __len__(self):
        return self.len


def get_loader(config, shuffle = True):
    """Load DataLoader"""

    dataset = MSADataset(config)
    config.data_len = len(dataset)

    def collate_fn(batch):
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x['data'][0][0].shape[0], reverse=True)
        
        index = []
        for sample in batch :
            index.append(sample['index']) 
        index = torch.LongTensor(index)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        labels = torch.cat([torch.from_numpy(sample['data'][1]) for sample in batch], dim = 0)
        sentences = pad_sequence([torch.LongTensor(sample['data'][0][0]) for sample in batch], padding_value = PAD)
        visual = pad_sequence([torch.FloatTensor(sample['data'][0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample['data'][0][2]) for sample in batch])
        segment = [sample['data'][2] for sample in batch]

        ## BERT-based features input prep
        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer
        bert_details = []
        text_list = []
        for sample in batch:
            text = " ".join(sample['data'][0][3])
            text_list.append(sample['data'][0][3])
            if config.base_model != 'magbert_model': NEW_SENT_LEN = SENT_LEN + 2
            else: NEW_SENT_LEN = SENT_LEN
            encoded_bert_sent = bert_tokenizer.encode_plus(text, max_length=NEW_SENT_LEN, add_special_tokens=True, pad_to_max_length=True)
            bert_details.append(encoded_bert_sent)
        
        batch_sentence_vector = []
        for sample in batch:
            sentence_vector = []
            for word in sample['data'][0][3]:
                word_id = dataset.word2id[word]
                sentence_vector.append(dataset.pretrained_emb[word_id])
            batch_sentence_vector.append(torch.stack(sentence_vector))

        sentences_vector = pad_sequence(batch_sentence_vector)

        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths_emb = torch.LongTensor([sample['data'][0][0].shape[0] for sample in batch])
        lengths = torch.LongTensor([sample['data'][0][0].shape[0] for sample in batch])
        
        sample_data = {
            'index': index,
            'raw_text': text_list,
            'text': to_gpu(sentences_vector, gpu_id = config.gpu_id),
            'audio': to_gpu(acoustic, gpu_id = config.gpu_id),
            'visual': to_gpu(visual, gpu_id = config.gpu_id),
            'labels_classify': to_gpu(change_to_classify(labels, config), gpu_id = config.gpu_id).squeeze(),
            'labels': to_gpu(labels, gpu_id = config.gpu_id).squeeze(),
            'lengths_emb': to_gpu(lengths_emb, gpu_id = config.gpu_id),
            'lengths': to_gpu(lengths, gpu_id = config.gpu_id),
            'bert_sentences': to_gpu(bert_sentences, gpu_id = config.gpu_id),
            'bert_sentence_att_mask': to_gpu(bert_sentence_att_mask, gpu_id = config.gpu_id),
            'bert_sentence_types': to_gpu(bert_sentence_types, gpu_id = config.gpu_id),
            'segment': segment,
        }

        return sample_data

    data_loader = DataLoader(
        dataset = dataset,
        batch_size = config.batch_size,
        shuffle = shuffle,
        collate_fn = collate_fn,
        drop_last = True,
    )

    return data_loader, len(dataset)