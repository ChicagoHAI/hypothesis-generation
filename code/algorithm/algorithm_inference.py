import argparse
import time
import pickle
import sys
import os
import math
import json

import random
import torch
import numpy as np

code_repo_path = os.environ.get("CODE_REPO_PATH")
sys.path.append(f'{code_repo_path}/code/')
from tasks import BaseTask
from prompt import BasePrompt
from data_loader import get_data
from utils import LLMWrapper, set_seed, get_results, GPT_MODELS, VALID_MODELS
from summary_information import dict_to_summary_information

from inference import INFERENCE_DICT

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def setup_LLM(args):
    api = LLMWrapper(args.model, 
                     path_name=args.model_path,
                     use_cache=args.use_cache)
    return api

def setup(args, seed, api):
    set_seed(seed)
    train_data, test_data, val_data = get_data(args)
    prompt_class = BasePrompt(BaseTask(args.task))
    inference_class = INFERENCE_DICT[args.inference_style](api, prompt_class, train_data)

    # return api, test_data, inference_class
    if args.use_valid:
        print('use valid')
        return val_data, inference_class
    print('use test')
    return test_data, inference_class


def parse_args():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--seeds', nargs='+', type=int, help='Random seed.')
    # TODO: config path instead of task name
    parser.add_argument('--task', type=str, choices=['shoe',
                                                     'hotel_reviews',
                                                     'headline_binary',
                                                     'retweet'
                                                     ], help='task to run')
    parser.add_argument('--model', type=str, default='claude_2', choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--model_path', type=str, default=None, help="Path for loading models locally.")
    # argument for using api cache, default true (1)
    parser.add_argument('--use_cache', type=int, default=1, help='Use cache for API calls.')
    
    parser.add_argument('--verbose', type=bool, default=True, help='Print more information.')
    parser.add_argument('--use_system_prompt', type=bool, default=True, help="Use instruction as system prompt.")

    # generation specific arguments
    parser.add_argument('--num_train', type=int, default=25, help='Number of training examples.')
    parser.add_argument('--num_test', type=int, default=100, help='Number of testing examples')
    parser.add_argument('--num_val', type=int, default=10, help='Number of validation examples')

    parser.add_argument('--use_valid', type=bool, default=False, help="Whether to use valid set as the testing set")
    # inference specific arguments
    parser.add_argument('--inference_style', type=str, choices=INFERENCE_DICT.keys(), help='types of inference methods')
    parser.add_argument('--k', type=int, default=1, help='Number of hypotheses to use')

    # knn specific arguments
    parser.add_argument('--knn_hypotheses', type=int, default=0, help="Number of hypotheses to choose from during KNN")
    parser.add_argument('--knn_num_examples', type=int, default=0, help='Number of examples per hypotheses to use for KNN')
    parser.add_argument('--knn_threshold', type=float, default=0.0, help='Threshold value for similarity matrix. If it is higher than these, then hypotheses are not selected')
    parser.add_argument('--add_examples', type=bool, default=False, help='Whether to add examples at the inference step (when we use KNN with separate steps).')
    parser.add_argument('--example_only_selection', type=bool, default=False, help='Whether to use only examples at the selection step (when we use KNN with separate steps).')
    parser.add_argument('--generate_prob', type=bool, default=False, help="Output probabilities.")
    # file specific arguments
    parser.add_argument('--use_ood_reviews', type=str, default="None", help="Use out-of-distribution hotel reviews.")
    parser.add_argument('--hypothesis_file', type=str, default=None, help="The file from which the hypotheses are loaded")
    args = parser.parse_args()

    assert args.hypothesis_file is not None, f'The hypothesis file must be provided'

    return args

def main():
    # set up tools
    start_time = time.time()
    args = parse_args()
    seeds = args.seeds
    accuracy_all = []
    f1_all = []
    dict = load_dict(args.hypothesis_file)
    hyp_bank = {}
    for hypothesis in dict:
        hyp_bank[hypothesis] = dict_to_summary_information(dict[hypothesis])

    assert args.knn_hypotheses <= len(hyp_bank), f'The number of hypotheses chosen in KNN must be less than the total number of hypotheses'
    api = setup_LLM(args)
    for seed in seeds:
        args.current_seed = seed
        test_data, inference_class = setup(args, seed, api)

        pred_list, label_list = inference_class.run_inference_final(args, test_data, hyp_bank)

        if args.task == 'shoe':
            accuracy = sum([1 if pred_list[i] == label_list[i] else 0 for i in range(len(pred_list))]) / len(pred_list)
            accuracy_all.append(accuracy)
            print(f"Accuracy for seed {seed}: {accuracy}")
        else:
            if args.inference_style == 'upperbound':
                continue
            tp, tn, fp, fn = get_results(args, pred_list, label_list)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

            accuracy_all.append(accuracy)
            f1_all.append(f1)

            print(f"Accuracy for seed {seed}: {accuracy}")
            print(f"F1 for seed {seed}: {f1}")

        # print the wrong indices
        wrong_indices = [i for i in range(len(pred_list)) if pred_list[i] != label_list[i]]
        print(f"Wrong indices: {wrong_indices}")

    print(f"Averaged accuracy: {sum(accuracy_all)/len(seeds)}")
    print(f"Averaged F1: {sum(f1_all)/len(seeds)}")
    
    # print experiment info
    print(f'Total time: {time.time() - start_time} seconds')
    # if api.model in GPT_MODELS:
    #     print(f'Estimated cost: {api.api.session_total_cost()}')

if __name__ == '__main__':
    main()