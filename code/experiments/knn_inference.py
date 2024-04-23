import pickle
import random
import time
import argparse
import sys
import os
import json 

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")

sys.path.append(f'{code_repo_path}/code')

from utils import LLMWrapper, SummaryInformation, set_seed, get_num_examples, extract_label, create_directory, GPT_MODELS, VALID_MODELS
from data_loader import get_data
from prompt import PROMPT_DICT
from tasks import TASKS


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

def load_hypotheses(args):
    if args.hypothesis_type == 'init':
        file_path = f'{args.hypothesis_dir}/hypotheses_training_sample_0.json'
    elif args.hypothesis_type == 'final':
        file_path = f'{args.hypothesis_dir}/hypotheses_training_sample_final.json'
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def top_k_hypotheses(args, hypotheses):
    return sorted(hypotheses, key=lambda x: hypotheses[x].reward, reverse=True)

def knn_inference(args, api, prompt_class, test_data, hypotheses):
    num_examples = get_num_examples(test_data)
    results = []
    for i in range(num_examples):
        print('********** Example', i, '**********')
        label = test_data['label'][i]
        prompt = prompt_class.knn_inference(hypotheses, test_data, i)
        response = api.generate(prompt)
        pred = extract_label(args.task, response)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Label: {label}")
        print(f"Prediction: {pred}")
        results.append({
            'prompt': prompt,
            'response': response,
            'label': label,
            'pred': pred
        })
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Run hypothesis inference experiment')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--task', type=str, default='shoe', choices=['shoe', 'headline_binary'], help='Task')
    parser.add_argument('--model', type=str, default='claude_2', choices=VALID_MODELS)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--hypothesis_type', type=str, default='final', choices=['final', 'init'], help='Type of hypotheses to use')
    parser.add_argument('--few_shot', type=int, default=5, help='Number of few shots.')
    parser.add_argument('--num_train', type=int, default=0, help='Number of training examples')
    parser.add_argument('--hypothesis_dir', type=str, default=None, help="Specifies the directory where the hypotheses are located")
    args = parser.parse_args()

    
    args = parser.parse_args()

    if args.task == 'shoe':
        args.hypotheses_dir = f'{code_repo_path}/outputs_{args.message}/{args.task}/{args.generation_model}_{args.seed}_None/' 
    

    return args


def main():
    # set up tools
    start_time = time.time()
    args = parse_args()
    set_seed(args)
    train_data, test_data = get_data(args.task, args.num_train, args.num_test)
    api = LLMWrapper(args.model)
    prompt_class = PROMPT_DICT[args.task]()
    #create_directory(args.output_folder)

    # load generated hypotheses
    x = load_hypotheses(args)
    hypotheses = {}
    for hyp in x.keys():
        hypotheses[hyp] = SummaryInformation(**x[hyp])

    
    top_k_hyp = top_k_hypotheses(args, hypotheses)
    top_k_hyp = top_k_hyp[:args.k]
    top_k_hyp_dict = {}
    for hyp in top_k_hyp:
        if len(hypotheses[hyp].correct_examples) > args.few_shot:
            top_k_hyp_dict[hyp] = random.sample(hypotheses[hyp].correct_examples, args.few_shot)
        else:
            top_k_hyp_dict[hyp] = hypotheses[hyp].correct_examples

    # inference: use the summary with the largest accuracy
    results = knn_inference(args, api, prompt_class, test_data, top_k_hyp_dict)
    
    
    # save results
    compute_accuracy(results)

    # print time and estimated cost
    print(f'Time: {time.time() - start_time} seconds')
    if api.model in GPT_MODELS:
        print(f'Estimated cost: {api.api.session_total_cost()}')
    


if __name__ == '__main__':
    main()