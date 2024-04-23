import pickle
import random
import time
import argparse
import sys
import os

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")

sys.path.append(f'{code_repo_path}/code')

from utils import LLMWrapper, set_seed, get_num_examples, extract_label, create_directory, GPT_MODELS, VALID_MODELS
from data_loader import get_data
from prompt import PROMPT_DICT
from tasks import TASKS


def compute_accuracy(results):
    labels = [result['label'] for result in results]
    preds = [result['pred'] for result in results]
    acc = sum([1 if labels[i] == preds[i] else 0 for i in range(len(labels))]) / len(labels)
    print(f'Accuracy: {acc}')


def inference_with_best_hypothesis(args, api, prompt_class, test_data, best_hypothesis, train_data=None):
    num_examples = get_num_examples(test_data)
    results = []
    for i in range(num_examples):
        print('********** Example', i, '**********')
        label = test_data['label'][i]
        prompt_input = prompt_class.new_inference_without_reasoning(best_hypothesis, test_data, i, few_shot=args.few_shot, train_data=train_data)
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


def inference_with_top_k_hypothesis(args, api, prompt_class, test_data, top_k_hypotheses, train_data=None):
    num_examples = get_num_examples(test_data)
    results = []
    for i in range(num_examples):
        print('********** Example', i, '**********')
        label = test_data['label'][i]
        preds = {}
        for hypothesis, summary_info in top_k_hypotheses:
            # print(f"Hypothesis: {hypothesis}, Accuracy: {summary_info.acc}")

            prompt_input = prompt_class.new_inference_without_reasoning(hypothesis, test_data, i, few_shot=args.few_shot, train_data=train_data)
            response = api.generate(prompt_input)
            pred = extract_label(args.task, response)

            preds[pred] = preds.get(pred, 0) + summary_info.acc

        # get the prediction with the highest weight
        pred = max(preds, key=preds.get)

        print(f"Preds: {preds}")
        
        print(f"Label: {label}")
        print(f"Prediction: {pred}")
        results.append({
            'label': label,
            'pred': pred
        })

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Run hypothesis inference experiment')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--task', type=str, default='shoe', choices=['shoe', 
                                                                     'retweet',
                                                                     ], help='Task')
    parser.add_argument('--inference_model', type=str, default='claude_2', choices=VALID_MODELS)
    parser.add_argument('--generation_model', type=str, default='claude_2', choices=VALID_MODELS)
    parser.add_argument('--message', type=str, default=None, help='Message for the experiment')
    parser.add_argument('--num_test', type=int, default=-1)
    parser.add_argument('--hypotheses_type', type=str, default='final', choices=['final', 'init'], help='Type of hypotheses to use')

    parser.add_argument('--strategy', type=str, default='best')
    parser.add_argument('--few_shot', type=int, default=0, help='Number of few shots.')
    parser.add_argument('--num_train', type=int, default=0, help='Number of training examples')
    parser.add_argument('--k', type=int, default=1, help='Number of hypotheses to use (top k)')

    args = parser.parse_args()

    if args.task in ['shoe', 'retweet']:
        args.hypotheses_dir = f'{code_repo_path}/outputs_{args.message}/{args.task}/{args.generation_model}_{args.seed}_None/' 
    args.output_folder = f'{code_repo_path}/outputs_{args.message}/{args.task}/inference_model_{args.inference_model}_generation_model_{args.generation_model}'

    return args


def main():
    # set up tools
    start_time = time.time()
    args = parse_args()
    set_seed(args)
    train_data, test_data = get_data(args.task, args.num_train, args.num_test)
    api = LLMWrapper(args.inference_model)
    prompt_class = PROMPT_DICT[args.task]()
    create_directory(args.output_folder)

    # load generated hypotheses
    if args.hypotheses_type == 'init':
        with open(f"{args.hypotheses_dir}/hypotheses_0.pkl", "rb") as f:
            hypotheses = pickle.load(f)
    elif args.hypotheses_type == 'final':
        with open(f"{args.hypotheses_dir}/hypotheses_final.pkl", "rb") as f:
            hypotheses = pickle.load(f)
    else:
        raise ValueError('Invalid hypotheses type: ' + args.hypotheses_type)

    # inference: use the summary with the largest accuracy
    if args.strategy == 'best':
        best_hypothesis_item = max(hypotheses.items(), key=lambda x: x[1].acc)
        best_hypothesis = best_hypothesis_item[0]
        results = inference_with_best_hypothesis(args, api, prompt_class, test_data, best_hypothesis, train_data=train_data)
        compute_accuracy(results)
    elif args.strategy == 'top_k':
        top_k_hypotheses = sorted(hypotheses.items(), key=lambda x: x[1].acc, reverse=True)[:args.k]
        results = inference_with_top_k_hypothesis(args, api, prompt_class, test_data, top_k_hypotheses, train_data=train_data)
        compute_accuracy(results)
    else:
        raise ValueError('Invalid strategy: ' + args.strategy)
    
    # save results
    with open(f"{args.output_folder}/{args.seed}_hypotheses{args.hypotheses_type}_test{args.num_test}_strategy{args.strategy}_few_shot{args.few_shot}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # print time and estimated cost
    print(f'Time: {time.time() - start_time} seconds')
    if api.model in GPT_MODELS:
        print(f'Estimated cost: {api.api.session_total_cost()}')
    


if __name__ == '__main__':
    main()