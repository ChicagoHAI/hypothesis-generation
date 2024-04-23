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
from prompt import PROMPT_DICT
from data_loader import get_data
from utils import LLMWrapper, set_seed, create_directory, get_num_examples, extract_label, GPT_MODELS, VALID_MODELS
from summary_information import SummaryInformation, dict_to_summary_information

from generation import GENERATION_DICT
from inference import INFERENCE_DICT
from replace import REPLACE_CHOICES, Replace
from update import UPDATE_DICT

def load_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def setup_LLM(args):
    api = LLMWrapper(args.model)
    return api

def setup(args, seed, api):
    set_seed(seed)
    train_data, _, _ = get_data(args)
    # api = LLMWrapper(args.model)
    prompt_class = PROMPT_DICT[args.task]()
    inference_class = INFERENCE_DICT[args.inference_style](api, prompt_class, train_data)
    generation_class = GENERATION_DICT[args.generation_style](api, prompt_class, inference_class)
    replace_class = Replace()
    update_class = UPDATE_DICT[args.update_style](generation_class, inference_class, replace_class)

    return train_data, update_class, inference_class, generation_class

def parse_args():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--seeds', nargs='+', type=int, help='Random seed.')
    parser.add_argument('--task', type=str, choices=['shoe',
                                                     'hotel_reviews',
                                                     'headline_binary',
                                                     'retweet'
                                                     ], help='task to run')
    parser.add_argument('--model', type=str, default='claude_2', choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--verbose', type=bool, default=True, help='Print more information.')
    parser.add_argument('--use_system_prompt', type=bool, default=True, help="Use instruction as system prompt.")
    # initialization specific arguments
    parser.add_argument('--num_init', type=int, default=25, help='Number of examples to use for initializing hypotheses.')
    parser.add_argument('--init_hypotheses_per_batch', type=int, default=5, help='Number of hypotheses to generate per batch')
    parser.add_argument('--init_batch_size', type=int, default=5, help='Batch size to generate hypotheses')

    # generation specific arguments
    parser.add_argument('--num_train', type=int, default=25, help='Number of training examples.')
    parser.add_argument('--num_test', type=int, default=25, help='Number of testing examples.')
    parser.add_argument('--num_val', type=int, default=25, help='Number of validation examples.')
    parser.add_argument('--save_every_n_examples', type=int, default=100, help='Save hypotheses every n examples.')
    parser.add_argument('--k', type=int, default=-1, help='The number of hypotheses checked per sample during training.')
    parser.add_argument('--max_num_hypotheses', type=int, default=20, help='Maximum number of hypotheses to keep in the hypotheses bank.')
    parser.add_argument('--num_hypotheses_to_update', type=int, default=5, help='Number of lowest-ranking hypotheses to update once we reach the maximum number of hypotheses.')
    parser.add_argument('--update_hypotheses_per_batch', type=int, default=5, help='Number of hypotheses to generate per prompt.') 
    parser.add_argument('--update_batch_size', type=int, default=5, help='Number of examples to use per prompt.')
    parser.add_argument('--generation_style', type=str, choices=GENERATION_DICT.keys(), help='types of generation methods')
    parser.add_argument('--generate_prob', type=bool, default=False, help="Output probabilities.")
    # reward specific arguments
    parser.add_argument('--alpha', type=float, default=5e-1, help='Exploration parameter.')

    # update specific arguments
    parser.add_argument('--update_style', type=str, choices=UPDATE_DICT.keys(), help='types of update')
    parser.add_argument('--num_wrong_to_add_bank', type=int, default=0, help='The number of hypotheses the sample it must be added the wrong examples bank. Must be greater than 1')
    parser.add_argument('--num_wrong_scale', type=float, default=0.8, help='Scale for dynamic num_wrong_to_add_bank')
    parser.add_argument('--only_best_hypothesis', type=bool, default=False, help='If only the best hypothesis should be added in the newly generated hypotheses of the batch')

    # inference specific arguments
    parser.add_argument('--inference_style', type=str, choices=INFERENCE_DICT.keys(), help='types of inference methods')

    # replace specific arguments
    parser.add_argument('--replace_style', type=str, choices=REPLACE_CHOICES, help='types of replace methods')

    # KNN specific arguments
    parser.add_argument('--knn_num_examples', type=int, default=0, help='Number of examples per hypotheses to use for KNN')

    # Restart specific arguments
    parser.add_argument('--old_hypothesis_file', type=str, default=None, help='Previously generated hypotheses to restart generation from')

    parser.add_argument('--sample_num_to_restart_from', type=int, default=-1, help='Sample number to resume from')
    parser.add_argument('--epoch_to_start_from', type=int, default=0, help='Epoch number to start from. When restarting, this should be > 1.')
    parser.add_argument('--current_epoch', type=int, default=-1, help='To keep track of current epoch, DO NOT CHANGE THIS.')

    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs to run the algorithm")
    parser.add_argument('--output_folder', type=str, default=None, help="Specifies the output path")
    args = parser.parse_args()
    if args.output_folder is None:
        args.output_folder = f'{code_repo_path}/print_log_{args.message}/{args.task}/{args.model}_{args.seed}_{args.hotel_inference_prompt}/'

    assert args.num_init <= args.num_train, f'Number of initialization examples ({args.num_init}) should be less than or equal to the number of training examples ({args.num_train}).'

    return args

def main():
    # set up tools
    start_time = time.time()
    args = parse_args()

    if (args.epoch_to_start_from) > 0 and (args.sample_num_to_restart_from) == -1:
        raise ValueError("When restarting from epoch > 0, you have to specify a sample_num_to_restart_from.")
    
    seeds = args.seeds
    create_directory(args.output_folder)
    api = setup_LLM(args)
    print(args.only_best_hypothesis)

    for seed in seeds:
        args.current_seed = seed
        train_data, update_class, inference_class, generation_class = setup(args, seed, api)
        hypotheses_bank = {}
        if args.old_hypothesis_file is None:
            args.current_epoch = 0
            hypotheses_bank = generation_class.initialize_hypotheses(args)
            update_class.save_to_json(args, f"{args.num_init}_seed_{args.current_seed}", hypotheses_bank)
        else:
            dict = load_dict(args.old_hypothesis_file)
            for hypothesis in dict:
                hypotheses_bank[hypothesis] = dict_to_summary_information(dict[hypothesis])

        for epoch in range(args.epoch_to_start_from,args.epoch_to_start_from+args.num_epochs):
            args.current_epoch = epoch
            hypotheses_bank = update_class.update(args, hypotheses_bank)
            update_class.save_to_json(args, f"final_seed_{seed}_epoch_{args.current_epoch}", hypotheses_bank)

        
        
    # print experiment info
    print(f'Total time: {time.time() - start_time} seconds')
    if api.model in GPT_MODELS:
        print(f'Estimated cost: {api.api.session_total_cost()}')

if __name__ == '__main__':
    main()
