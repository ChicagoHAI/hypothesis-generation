import argparse
import time
import pickle
import sys
import os
import math
import logging 
import random
import torch 
import numpy as np

code_repo_path = os.environ.get("CODE_REPO_PATH")
sys.path.append(f'{code_repo_path}/code/')
from prompt import PROMPT_DICT
from data_loader import get_data
from utils import LLMWrapper, SummaryInformation, set_seed, create_directory, get_num_examples, extract_label, GPT_MODELS, VALID_MODELS
from hypotheses_initialization import initialize_hypotheses, extract_hypotheses

def compute_accuracy(results):
        
    labels = [result['label'] for result in results]
    preds = [result['pred'] for result in results]
    a = 0
    x = []
    for label, pred in zip(labels, preds):
        if pred == "other":
            a += 1
        if pred == label:
            x.append(1)
        else:
            x.append(0)
    acc = sum(x)/len(x)
    print("non-safety mode record:", len(x)-a)
    print(f'Accuracy: {acc}')
    return acc 

def few_shot(args, prompt_class, api, train_data, test_data):
    num_examples = get_num_examples(test_data)
    results = []
    for i in range(args.num_test):
        print('********** Example', i, '**********')
        label = test_data['label'][i]
        prompt_input = prompt_class.few_shot_baseline(train_data, args.few_shot, test_data, i)
        response = api.generate(prompt_input)
        pred = extract_label(args.task, response)
        print(f"Prompt: {prompt_input}")
        print(f"Response: {response}")
        print(f"Label: {label}")
        print(f"Prediction: {pred}")
        results.append({
            'prompt': prompt_input,
            'response': response,
            'label': label,
            'pred': pred
        })
        
    return results

def preprocess(train_data, k):
    num_examples = get_num_examples(train_data)
    flag = True
    data = {}
    for key in train_data.keys():
        data[key] = []

    for i in range(k-1):
        if train_data['label'][i] != train_data['label'][i+1]:
            flag = False
            break 

    if flag:
        for j in range(k-1):
            for key in train_data.keys():
                data[key].append(train_data[key][j])
        
        for i in range(k+1, num_examples):
            if train_data['label'][i] != data['label'][-1]:
                for key in train_data.keys():
                    data[key].append(train_data[key][i])
                break 
        return data 
    else:
        for j in range(k):
            for key in train_data.keys():
                data[key].append(train_data[key][j])
        return data 


def parse_args():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--seeds', nargs='+', type=int, help='Random seed.')
    parser.add_argument('--task', type=str, choices=['shoe',
                                                     'hotel_reviews',
                                                     'headline_binary'
                                                     ], help='task to run')
    parser.add_argument('--model', type=str, default='claude_2', choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--few_shot', type=int, default=0, help='Number of few-shot examples')
    parser.add_argument('--num_train', type=int, default=100, help='Number of training examples')
    parser.add_argument('--num_test', type=int, default=100, help='Number of test examples')

    args = parser.parse_args()
    return args

def main():
    # set up tools
    start_time = time.time()
    args = parse_args()
    seeds = list(args.seeds)
    acc = []
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print('Getting data ...')
        api = LLMWrapper(args.model)
        prompt_class = PROMPT_DICT[args.task]()


        train_data, test_data = get_data(args.task, args.num_train, args.num_test)
        if args.few_shot > 0:
            train_data = preprocess(train_data, args.few_shot)

        results = few_shot(args, prompt_class, api, train_data, test_data)
        acc.append(compute_accuracy(results))
        
        print(f'Time: {time.time() - start_time} seconds')
        #if api.model in GPT_MODELS:
            #print(f'Estimated cost: {api.api.session_total_cost()}')

    print(sum(acc)/len(acc))

if __name__ == '__main__':
    main()