#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import pandas as pd
import numpy as np
import re
import tarfile
import pickle as pickle
from tqdm import tqdm
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from loss import *

import wandb

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

#wandb.init(project='p-stage-2-competition', entity='newspring97')
#wandb.run.name = 'skt-kobert'#cfg['run_name']

device = torch.device("cuda:0")


# kobert 불러오기
bertmodel, vocab = get_pytorch_kobert_model()


# # Preprocessing
def load_data(dataset_dir):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    dataset = preprocessing_dataset(dataset, label_type)
    return dataset

def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
    return out_dataset

dataset_path = r"/opt/ml/input/data/train/train.tsv" #TODO: dataset 추가하기

dataset = load_data(dataset_path)

dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']

#train, vali = train_test_split(dataset, test_size=0.2, random_state=42)
dataset[['sentence','label']].to_csv("/opt/ml/input/data/train/train_train.txt", sep='\t', index=False)
#vali[['sentence','label']].to_csv("/opt/ml/input/data/train/train_vali.txt", sep='\t', index=False)

dataset_train = nlp.data.TSVDataset("/opt/ml/input/data/train/train_train.txt", field_indices=[0,1], num_discard_samples=1)
#dataset_vali = nlp.data.TSVDataset("/opt/ml/input/data/train/train_vali.txt", field_indices=[0,1], num_discard_samples=1)


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


max_len = 128
batch_size = 32
warmup_ratio = 0.01
num_epochs = 10
max_grad_norm = 1
log_interval = 50
learning_rate = 5e-5


# In[12]:


data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
#data_vali = BERTDataset(dataset_vali, 0, 1, tok, max_len, True, False)


# In[13]:


train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
#vali_dataloader = torch.utils.data.DataLoader(data_vali, batch_size=batch_size, num_workers=5)


# # Classification

# In[14]:


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 42,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]



optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = LabelSmoothingLoss(smoothing=0.2)


# In[19]:


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)


# In[20]:


scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# In[21]:


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


# In[24]:


for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    best_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    '''
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(vali_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    '''
    torch.save(model.state_dict(), "/opt/ml/model/model_kobert_0_2_epoch_10.pt")


# # Predict

# In[25]:


dataset_path = r"/opt/ml/input/data/test/test.tsv"

dataset = load_data(dataset_path)

dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']

dataset[['sentence','label']].to_csv("/opt/ml/input/data/test/test.txt", sep='\t', index=False)


# In[26]:


dataset_test = nlp.data.TSVDataset("/opt/ml/input/data/test/test.txt", field_indices=[0,1], num_discard_samples=1)

data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)


# In[27]:


model.load_state_dict(torch.load("/opt/ml/model/model_kobert_0_2_epoch_10.pt"))

model.eval()

Predict = []

for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length = valid_length
    label = label.long().to(device)
    out = model(token_ids, valid_length, segment_ids)
    _, predict = torch.max(out,1)
    Predict.extend(predict.tolist())


output = pd.DataFrame(Predict, columns=['pred'])
output.to_csv('/opt/ml/submission_kobert_0_2_epoch_10.csv', index=False)

