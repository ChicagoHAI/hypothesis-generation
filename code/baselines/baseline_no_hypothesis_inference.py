# call LLM to predict label without taking in a hypothesis
# zero-shot learning, with instruction in the prompt

import argparse
import time
import sys
import os

code_repo_path = os.environ.get("CODE_REPO_PATH")
if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable CODE_REPO_PATH not set.")

sys.path.append(f'{code_repo_path}/code/')
from tasks import BaseTask
from utils import LLMWrapper, get_num_examples, extract_label, create_directory, set_seed, VALID_MODELS, GPT_MODELS
from data_loader import get_data
from prompt import BasePrompt


def compute_accuracy(results):
    labels = [result['label'] for result in results]
    preds = [result['pred'] for result in results]
    safety_mode = 0
    x = []
    for label, pred in zip(labels, preds):
        if pred == "other":
            safety_mode += 1
        if pred == label:
            x.append(1)
        else:
            x.append(0)
    acc = sum(x)/len(x)
    print("non-safety mode record:", len(x)-safety_mode)
    print(f'Accuracy: {acc}')
    return acc 


def few_shot(args, api, train_data, test_data, prompt_class):
    """
        Given one hyothesis and a dataset, return the accuracy of the hypothesis on the dataset.
    """
    results = []
    for i in range(args.num_test):
        prompt_input = prompt_class.few_shot_baseline(train_data, args.few_shot, test_data, i, model_name=args.inference_model)
        response = api.generate(prompt_input,inst_in_sys=args.use_system_prompt)
        print(f'********** Example {i} **********')
        pred = extract_label(args.task, response)
        label = test_data['label'][i]

        # print(f"Prompt: {prompt_input}")
        print(f"Response: {response}")
        print(f"Label: {label}")
        print(f"Prediction: {pred}")
        results.append({
            'prompt': prompt_input,
            'response': response,
            'label': label,
            'pred': pred
        })
        print('**********************************')
        
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

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--task', type=str, choices=['headline_binary',
                                                     'shoe',
                                                     'retweet',
                                                     'hotel_reviews'
                                                     ], help='task to run')
    parser.add_argument('--inference_model', type=str, default='Mixtral-8x7B', choices=VALID_MODELS, help='Model to use at inference time.')
    parser.add_argument('--message', type=str, default='no_message', help='A note on the experiment setting.')
    parser.add_argument('--num_test', type=int, default=100, help='Number of test examples.')
    parser.add_argument('--num_train', type=int, default=100, help='Number of train examples.')
    parser.add_argument('--num_val', type=int, default=100, help='Number of validation examples.')
    parser.add_argument('--few_shot', type=int, default=3, help='Number of examples to use as context.')
    parser.add_argument('--use_system_prompt', type=bool, default=True, help="Use instruction as system prompt.")
    parser.add_argument('--use_ood_reviews', type=str, default="None", help="Use out-of-distribution hotel reviews.")
    parser.add_argument('--model_path', type=str, default=None, help="Path for loading models locally.")
    
    # argument for using api cache, default true (1)
    parser.add_argument('--use_cache', type=int, default=1, help='Use cache for API calls.')
    
    args = parser.parse_args()

    return args


def main():
    start_time = time.time()
    args = parse_args()
    set_seed(args)

    prompt_class = BasePrompt(BaseTask(args.task))
    api = LLMWrapper(args.inference_model, 
                     path_name=args.model_path,
                     use_cache=args.use_cache)

    train_data, test_data, _ = get_data(args)

    if args.few_shot > 0:
        train_data = preprocess(train_data, args.few_shot)

    results = few_shot(args, api, train_data, test_data, prompt_class)
    test_accuracy = compute_accuracy(results)

    print('Test accuracy: ', test_accuracy)
    print('Total time (seconds): ', round(time.time() - start_time, 2))


if __name__ == '__main__':
    main()