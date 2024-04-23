from transformers import RobertaTokenizer
from transformers import RobertaConfig, RobertaModel
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
import torch
from torch.utils.data import DataLoader, random_split,Dataset

import time
import random
import pickle
import sys
import math
import argparse
import os
from tqdm import tqdm

import sys
sys.path.append('/home/haokunliu/past-interaction-learning/code')
from data_loader import get_train_data, get_test_data, get_data
from dicts import LABEL_DICT, PROMPT_NAME_DICT


class Seq_data(Dataset):
    def __init__(self, prompts, labels):
        self.prompts = prompts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx]
    
class RoBERTa_classif(nn.Module):
    def __init__(self,
                 RoBERTa,
                 num_classes,
                 num_layers=1,
                 device="cuda:0"):
        super().__init__()
        self.RoBERTa = RoBERTa
        self.hidden_size = 50265

        linear_relu_stack = []
        for i in range(num_layers-1):
            linear_relu_stack.append(nn.Linear(self.hidden_size,self.hidden_size))
            linear_relu_stack.append(nn.ReLU())

        self.linear_relu_stack = nn.ModuleList(linear_relu_stack)

        self.fc = nn.Linear(self.hidden_size,num_classes)
        self.num_layers = num_layers
        self.device = device

    def forward(self,x):
        out = self.RoBERTa(**x.to(self.device))
        out = out.logits
        out = out[:,0,:]

        if len(self.linear_relu_stack)>0:
            for _, layer in enumerate(self.linear_relu_stack):
                out = layer(out)

        out = self.fc(out)
        return out
    
def run_test(
         SEED=1,
         device="cuda:0",
         task_name="shoe",
         num_train=100,
         num_test=100,
         BATCH_SIZE=16,
         EPOCHS=10,
         LR=1e-4,
         NUM_LAYERS=1):

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cache_dir = "/net/scratch/haokunliu/llama_cache"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base",cache_dir=cache_dir)
    config = AutoConfig.from_pretrained("roberta-base",cache_dir=cache_dir)
    config.is_decoder = True
    RoBERTa = RobertaForCausalLM.from_pretrained("roberta-base", config=config, cache_dir=cache_dir)

    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    train_data, test_data = get_data(task_name=task_name,
                                    num_train=num_train,
                                    num_test=num_test)

    num_classes = len(LABEL_DICT[task_name])
    prompt_name = PROMPT_NAME_DICT[task_name]
    train_prompts = train_data[prompt_name]
    test_prompts = test_data[prompt_name]
        
    train_labels = process_labels(task_name,train_data['label'])
    test_labels = process_labels(task_name,test_data['label'])

    dataset_train = Seq_data(train_prompts, train_labels)
    dataloader_train = DataLoader(dataset_train,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=False
                                )

    model = RoBERTa_classif(RoBERTa,
                            num_classes=num_classes,
                            num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    train_acc = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        with tqdm(total=len(dataloader_train), desc=f"[Epoch {epoch+1:3d}/{EPOCHS}]") as pbar:
            for idx_batch, (prompts, y) in enumerate(dataloader_train):
                x = tokenizer(prompts, return_tensors="pt", padding=True)
                y = y.to(device)
                optimizer.zero_grad()

                output = model(x)
                loss = loss_fn(output.squeeze(),y)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.update(1)

            train_loss = running_loss/len(dataloader_train)
            losses.append(train_loss)

            pbar.set_postfix({'train loss': train_loss})

        model.eval()
        with torch.no_grad():
            eval_outputs = model(tokenizer(train_prompts,return_tensors="pt", padding=True).to(device))
            pred_labels = torch.argmax(eval_outputs,dim=1).cpu().numpy()
            acc = np.sum(pred_labels==train_labels)/len(train_labels)
            train_acc.append(acc)

    # train_data, test_data = get_data(task_name=task_name,
    #                                  num_train=num_train,
    #                                  num_test=num_test)

    # num_classes = len(LABEL_DICT[task_name])
    # prompt_name = PROMPT_NAME_DICT[task_name]
    # train_prompts = train_data[prompt_name]
    # test_prompts = test_data[prompt_name]

    # train_labels = process_labels(task_name,train_data['label'])
    # test_labels = process_labels(task_name,test_data['label'])
    
    # dataset_train = Seq_data(train_prompts, train_labels)
    # dataloader_train = DataLoader(dataset_train,
    #                           batch_size=BATCH_SIZE,
    #                           shuffle=False,
    #                           num_workers=0,
    #                           pin_memory=False
    #                          )

    # model = RoBERTa_classif(RoBERTa,
    #                       num_classes=num_classes,
    #                       num_layers=NUM_LAYERS).to(device)
    # optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    # loss_fn = nn.CrossEntropyLoss()
    
    # losses = []
    # train_acc = []

    # for epoch in range(EPOCHS):
    #     model.train()
    #     running_loss = 0
    #     with tqdm(total=len(dataloader_train), desc=f"[Epoch {epoch+1:3d}/{EPOCHS}]") as pbar:
    #         for idx_batch, (prompts, y) in enumerate(dataloader_train):
    #             x = tokenizer(prompts, return_tensors="pt", padding=True)
    #             y = y.to(device)
    #             optimizer.zero_grad()

    #             output = model(x)
    #             loss = loss_fn(output.squeeze(),y)
    #             running_loss += loss.item()
    #             loss.backward()
    #             optimizer.step()

    #             pbar.update(1)

    #         train_loss = running_loss/len(dataloader_train)
    #         losses.append(train_loss)

    #         pbar.set_postfix({'train loss': train_loss})
        
    #     model.eval()
    #     with torch.no_grad():
    #         eval_outputs = model(tokenizer(train_prompts,return_tensors="pt", padding=True).to(device))
    #         pred_labels = torch.argmax(eval_outputs,dim=1).cpu().numpy()
    #         acc = np.sum(pred_labels==train_labels)/len(train_labels)
    #         train_acc.append(acc)

    model.eval()
    with torch.no_grad():
        eval_outputs = model(tokenizer(test_prompts,return_tensors="pt", padding=True).to(device))
        pred_labels = torch.argmax(eval_outputs,dim=1).cpu().numpy()
        test_acc = np.sum(pred_labels==test_labels)/len(test_labels)

    return train_acc, test_acc