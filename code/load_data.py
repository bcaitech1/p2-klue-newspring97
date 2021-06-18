import pickle as pickle
import os
import pandas as pd
import torch

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset

def add_special_tok(texts, e1_starts, e1_ends, e2_starts, e2_ends):
  result = []
  for i in range(len(texts)):
    if e1_starts[i] < e2_starts[i]:
      new_text = texts[i][:e1_starts[i]] + ' [E1] ' + texts[i][e1_starts[i]:e1_ends[i]+1] + \
                ' [/E1] ' + texts[i][e1_ends[i]+1:e2_starts[i]] + ' [E2] ' + texts[i][e2_starts[i]:e2_ends[i]+1] + \
                ' [/E2] ' + texts[i][e2_ends[i]+1:]
    else:
      new_text = texts[i][:e2_starts[i]] + ' [E2] ' + texts[i][e2_starts[i]:e2_ends[i]+1] + \
                ' [/E2] ' + texts[i][e2_ends[i]+1:e1_starts[i]] + ' [E1] ' + texts[i][e1_starts[i]:e1_ends[i]+1] + \
                ' [/E1] ' + texts[i][e1_ends[i]+1:]
    result.append(new_text)
  return result

def preprocessing_dataset_with_entity(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence': add_special_tok(dataset[1], dataset[3], dataset[4], dataset[6], dataset[7]),
                              'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset

def load_data_with_entity(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset_with_entity(dataset, label_type)
  
  return dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      )
  return tokenized_sentences
