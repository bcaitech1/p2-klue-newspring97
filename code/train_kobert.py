import pickle as pickle
import os
import pandas as pd
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from load_data import *

import wandb
import argparse
import json
from importlib import import_module

from tokenization_kobert import KoBertTokenizer


def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train(cfg):
  seed_everything(cfg["seed"])
  # 1. Start a new run
  wandb.init(project='p-stage-2-competition', entity='newspring97')
  wandb.run.name = cfg['run_name']

  # load model and tokenizer
  MODEL_NAME = cfg['model_name']

  # KoBert용 tokenizer
  tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

  # load dataset
  train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  #dev_dataset = load_data("./dataset/train/dev.tsv")
  train_label = train_dataset['label'].values
  #dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(torch.cuda.is_available())

  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  #model_config = getattr(import_module("transformers"), cfg["model"] + "Config").from_pretrained(MODEL_NAME)
  model_config.num_labels = 42
  #model = getattr(import_module("transformers"), cfg["model"] + "ForSequenceClassification")(model_config)
  #model = AutoModelForSequenceClassification.from_config(model_config) 
  model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], config=model_config)
  model.parameters
  model.to(device)
  #wandb.watch(model)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=cfg["output_dir"],          # output directory
    save_total_limit=cfg["save_total_limit"],              # number of total save model
    save_steps=cfg["save_steps"],                 # model saving step
    num_train_epochs=cfg["num_train_epochs"],              # total number of training epochs
    learning_rate=cfg["learning_rate"],               # learning_rate
    per_device_train_batch_size=cfg["per_device_train_batch_size"],  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=cfg["warmup_steps"],                # number of warmup steps for learning rate scheduler
    weight_decay=cfg["weight_decay"],               # strength of weight decay
    logging_dir=cfg["logging_dir"],            # directory for storing logs
    logging_steps=cfg["logging_steps"],              # log saving step
    #evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training
                                # `steps`: Evaluate every `eval_steps`
                                # `epoch`: Evaluate every end of epoch
    #eval_steps = 500,            # evaluation step
    report_to=["wandb"],
    run_name = cfg["run_name"],
    seed=cfg["seed"],
    label_smoothing_factor=cfg["label_smoothing_factor"] if "label_smoothing_factor" in cfg.keys() else 0
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    #eval_dataset=RE_dev_dataset,             # evaluation dataset
    #compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()

def main(cfg_file):
  with open(cfg_file) as json_file:
    cfg = json.load(json_file)
    print(cfg)
  train(cfg)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Get config.json file for train.py')
  parser.add_argument('--config', dest='cfg', type=str, default='configs/config.json')

  args = parser.parse_args()
  main(args.cfg)
