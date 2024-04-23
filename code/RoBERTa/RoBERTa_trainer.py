import torch
import torch.nn as nn
from datasets import load_dataset
import argparse

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from huggingface_hub import HfFolder, notebook_login
import numpy as np
import sys
import os
code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")

sys.path.append(f'{code_repo_path}/code/')

import evaluate
from data_loader import get_data
from RoBERTa.dicts import LABEL_DICT, PROMPT_NAME_DICT, reverse_dict
from datasets import Dataset
import matplotlib.pyplot as plt

class Data_args(argparse.Namespace):
    def __init__(self, 
                 task,
                 num_train=100,
                 num_test=100,
                 num_val=100,
                 use_ood_reviews="None",
                 **kwargs):
        self.num_train = num_train
        self.num_test = num_test
        self.num_val = num_val
        self.task = task
        self.use_ood_reviews = use_ood_reviews

def process_labels(task,labels):
    numeric_labels = []
    label_dict = LABEL_DICT[task]
    for label in labels:
        numeric_labels.append(label_dict[label])
    return numeric_labels


def set_seed(seed):
    print(f"Setting seed to {seed}")
    import random
    import torch
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def prepare_data_for_trainer(task_name,num_train,num_test,num_val,use_ood_reviews="None"):
    data_args = Data_args(task_name,
                          num_train,
                          num_test,
                          num_val,
                          use_ood_reviews)
    train_data, test_data, val_data = get_data(data_args)

    train_labels = process_labels(task_name,train_data['label'])
    test_labels = process_labels(task_name,test_data['label'])
    val_labels = process_labels(task_name,val_data['label'])

    # The get_data function gets the label with their string values
    # Switch to numerical values
    train_data['label'] = train_labels
    test_data['label'] = test_labels
    val_data['label'] = val_labels

    # Construct HF dataset
    train_data = Dataset.from_dict(train_data)
    test_data = Dataset.from_dict(test_data)
    val_data = Dataset.from_dict(val_data)

    return train_data, test_data, val_data


def tokenize_data(model_id, task_name, num_train, num_test, num_val,use_ood_reviews="None"):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
    train_data, test_data, val_data = prepare_data_for_trainer(task_name, num_train, num_test, num_val,use_ood_reviews)

    def tokenize(batch):
        if task_name in ['retweet', 'headline_binary']:
            return tokenizer.batch_encode_plus(batch[PROMPT_NAME_DICT[task_name]], padding=True, truncation=True, max_length=512)
        else:
            return tokenizer(batch[PROMPT_NAME_DICT[task_name]], padding=True, truncation=True, max_length=512)
    
    train_dataset = train_data.map(tokenize, batched=True, batch_size=len(train_data))
    test_dataset = test_data.map(tokenize, batched=True, batch_size=len(test_data))
    val_dataset = val_data.map(tokenize, batched=True, batch_size=len(val_data))

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])  
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, test_dataset, val_dataset


def prepare_trainer(model_id, 
                    task_name, 
                    num_train, 
                    num_test, 
                    num_val,
                    seed, 
                    output_dir,
                    use_optimzer_and_scheduler=False,
                    use_ood_reviews="None",
                    **kwargs):
    set_seed(seed)
    # Create an id2label mapping
    id2label = reverse_dict(LABEL_DICT[task_name])

    # Update the model's configuration with the id2label mapping
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label})

    # For classification task, prepare accuracy as our evaluation metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Model
    model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)
    train_dataset, test_dataset, val_dataset = tokenize_data(model_id, task_name, num_train, num_test, num_val,use_ood_reviews)
    training_args = TrainingArguments(
        output_dir=output_dir,
        **kwargs
    )

    # Define optimizer and learning rate scheduler
    if use_optimzer_and_scheduler:
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['learning_rate'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95) 

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # optimizers=(optimizer, lr_scheduler) if use_optimzer_and_scheduler else None,
        compute_metrics=compute_metrics,
    )

    return trainer, train_dataset, test_dataset, val_dataset

