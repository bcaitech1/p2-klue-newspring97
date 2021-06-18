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


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


dataset_path = r"/opt/ml/input/data/test/test.tsv"

dataset = load_data(dataset_path)

dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']

dataset[['sentence','label']].to_csv("/opt/ml/input/data/test/test.txt", sep='\t', index=False)


# In[26]:
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)



dataset_test = nlp.data.TSVDataset("/opt/ml/input/data/test/test.txt", field_indices=[0,1], num_discard_samples=1)

data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)


# In[27]:


model.load_state_dict(torch.load("/opt/ml/model/model_kobert_nosmoothing.pt"))

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
output.to_csv('/opt/ml/submission_kobert_nosmoothing.csv', index=False)
