import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from datasets import Dataset, load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

import sys
import os
code_repo_path = os.environ.get("CODE_REPO_PATH")
sys.path.append(f'{code_repo_path}/code')
from data_loader import get_train_data, get_test_data, get_data
from dicts import LABEL_DICT, PROMPT_NAME_DICT, reverse_dict
from RoBERTa_trainer import prepare_trainer


def finetune_roberta(max_steps, per_device_batch_size, learning_rate, num_train, num_test, num_val, seed, task, wandb=False, output_dir='roberta_results'):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Weights & Biases
    if wandb:
        wandb.login()
        wandb.init(project="hypo-gen", name=f"seed{seed}-train{num_train}-unfreeze-retweet-roberta-lf{learning_rate}-add-optimizer-scheduler-warmup", reinit=True)

        # Log hyperparameters
        wandb.config.update({
            "max_steps": max_steps,
            "per_device_batch_size": per_device_batch_size,
            "learning_rate": learning_rate,
            "num_train": num_train,
            "num_test": num_test,
            "num_val": num_val,
            "seed": seed
        })

    args = {
        'num_train_epochs':20,
        'per_device_train_batch_size':per_device_batch_size,
        'per_device_eval_batch_size':per_device_batch_size,
        'learning_rate':learning_rate,
        'weight_decay':0.01,
        'warmup_steps':500,
        'logging_strategy':"steps",
        'logging_steps':10,
        'evaluation_strategy':"epoch",
        'save_strategy':"epoch",
        'save_total_limit':2,
        'load_best_model_at_end':True,
        'report_to': "wandb" if wandb else None,
        'output_dir': output_dir
    }

    prepare_trainer(model_id="roberta-base",
        task_name=task,
        num_train=num_train,
        num_test=num_test,
        num_val=num_val,
        seed=seed,
        **args)
    
    # Define trainer
    trainer, train_dataset, test_dataset, val_dataset = prepare_trainer(model_id="roberta-base",
        task_name=task,
        num_train=num_train,
        num_test=num_test,
        num_val=num_val,
        seed=seed,
        use_optimzer_and_scheduler=True,
        **args)

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(eval_results)

    if wandb:
        # save eval_results in wandb
        wandb.log(eval_results)
        # Finish Weights & Biases run
        wandb.finish()

    eval_acc = eval_results['eval_accuracy']
    return eval_acc


def main():
    max_steps=500
    per_device_batch_size=1
    num_test=100
    num_val=100
    learning_rate=2e-6

    eval_acc = {}
    for seed in [49, 50, 51]:
        eval_acc[seed] = {}
        for num_train in [1, 3, 6, 12, 25, 50, 75, 100]:
            finetune_roberta(max_steps, 
                             per_device_batch_size, 
                             learning_rate, 
                             num_train, 
                             num_test, 
                             num_val,
                             seed,
                             task='retweet',
                             wandb=True)
            eval_acc[seed][num_train] = eval_acc

    print(eval_acc)

