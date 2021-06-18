from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

import json


def inference_huggingface(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_logits = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            if 'token_type_ids' in data.keys():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
                )
            else:
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device)
                )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.append(result)
    return np.array(output_pred).flatten()


model_path_huggingface = [
    #TODO: 앙상블에 사용할 모델 목록
]

model_path_kobert = [
    #TODO: 앙상블에 사용할 모델 목록
]

