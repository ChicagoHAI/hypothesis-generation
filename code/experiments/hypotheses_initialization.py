# initialize hypotheses

import time
import random
import pickle
import sys
import argparse
import os

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")

sys.path.append(f'{code_repo_path}/code')

from utils import LLMWrapper, SummaryInformation, get_num_examples, extract_label, create_directory, set_seed, VALID_MODELS, GPT_MODELS
from prompt import PROMPT_DICT
from data_loader import get_train_data
from pprint import pprint


def extract_hypotheses(args, text):
    import re
    pattern = re.compile(r"\d+\.\s(.+?)(?=\d+\.\s|\Z)", re.DOTALL)
    print("Text provided", text)
    hypotheses = pattern.findall(text)
    if len(hypotheses) == 0:
        print("No hypotheses are generated.")
        return []

    for i in range(len(hypotheses)):
        hypotheses[i] = hypotheses[i].strip()

    return hypotheses[:args.num_hypotheses_per_init_prompt]


def initialize_hypotheses(args, api, train_data, prompt_class):
    assert args.num_examples_per_init_prompt > 0

    # get a key in the train_data
    key = list(train_data.keys())[0]
    assert len(train_data[key]) % args.num_examples_per_init_prompt == 0, f'Number of training examples ({len(train_data[key])}) is not divisible by number of examples per prompt ({args.num_examples_per_init_prompt}).'

    num_generations = len(train_data[key]) // args.num_examples_per_init_prompt
    hypotheses = []
    for i in range(num_generations):
        example_data = {}
        for key in train_data:
            example_data[key] = train_data[key][i * args.num_examples_per_init_prompt: (i + 1) * args.num_examples_per_init_prompt]

        if args.verbose: 
            print("**** hypothesis initialization ****")

        prompt_input = prompt_class.batched_learning_hypothesis_generation(example_data, args.num_hypotheses_per_init_prompt)

        if args.verbose: 
            print(f"Prompt: \n{prompt_input} \n")

        response = api.generate(prompt_input)

        if args.verbose: 
            print("prompt length: ", len(prompt_input))
            print(f"Response: \n{response} \n")
            print("response length: ", len(response))
            print("************************************")

        # extract the hypotheses from the response
        extracted_hypotheses = extract_hypotheses(args, response)
        hypotheses.extend(extracted_hypotheses)

    return hypotheses


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_train', type=int, default=25, help='Number of training examples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--task', type=str, choices=['shoe',
                                                     'binary_original_sst',
                                                     'hotel_reviews',
                                                     ], help='task to run')
    parser.add_argument('--model', type=str, default='chatgpt', choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--message', type=str, default='no_message', help='A note on the experiment setting.')
    parser.add_argument('--output_folder', type=str, default=None, help='Path to save the hypotheses.')
    parser.add_argument('--verbose', action='store_true', help='Print more information.')

    # initialization specific arguments
    parser.add_argument('--num_hypotheses_per_init_prompt', type=int, default=5, help='Number of hypotheses to generate per prompt.')
    parser.add_argument('--num_examples_per_init_prompt', type=int, default=5, help='Number of examples to use per prompt.')

    args = parser.parse_args()

    return args


def main():
    start_time = time.time()
    args = parse_args()
    set_seed(args)
    train_data = get_train_data(args)
    api = LLMWrapper(args.model)
    prompt_class = PROMPT_DICT[args.task]()
    create_directory(args.output_folder)

    # initialize hypotheses
    hypotheses = initialize_hypotheses(args, api, train_data, prompt_class)

    # save the hypotheses
    with open(f'{args.output_folder}/{args.model}_seed{args.seed}_train{args.num_train}_{args.num_hypotheses_per_init_prompt}hypotheses_per_prompt_{args.num_examples_per_init_prompt}examples_per_prompt.pkl', 'wb') as f:
        pickle.dump(hypotheses, f)
    print('Hypotheses saved:')
    pprint(hypotheses)

    # print experiment info
    print(f'Time: {time.time() - start_time} seconds')
    if api.model in GPT_MODELS:
        print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == '__main__':
    main()