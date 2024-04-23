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

from utils import LLMWrapper, SummaryInformation, extract_label, compute_binary_metrics, get_num_examples, create_directory, set_seed, VALID_MODELS, GPT_MODELS
from data_loader import get_test_data
from prompt import PROMPT_DICT
from tasks import TASKS


def get_best_summary(hypothesis, args):
    """
    Get the summary with the highest reward

    Args:
        hypothesis: a dictionary of summaries and their rewards

    Returns:
        summary_high_reward: the summary with the highest reward
    """
    # sort summaries by reward (third element in value)
    sorted_hyp = sorted(hypothesis.items(), key=lambda x: x[1].reward, reverse=True)
    summary_high_reward = sorted_hyp[0][0]
    if args.verbose: print('Initial best summary:', summary_high_reward)
    if args.task == 'diplomacy' and len(summary_high_reward) > 200:
        for i in range(1, 5):
            s = sorted_hyp[i][0]
            if len(s) < 200:
                summary_high_reward = s
                if args.verbose: print('Updated best summary:', summary_high_reward)
                break

    # filter out summaries that are too long
    sorted_filtered_hyp = []
    for s, v in sorted_hyp:
        if len(s) < 200:
            sorted_filtered_hyp.append((s, v))
    if args.verbose: print('Number of summaries after filtering:', len(sorted_filtered_hyp))

    return summary_high_reward, sorted_filtered_hyp


def get_best_relevant_summary(args, api, hypothesis, test_data, i, prompt):
    # go through the sorted hypotheses, find the first one that is relevant
    for s, _ in hypothesis[:min(10, len(hypothesis))]: # check 10 hypotheses at most
        prompt_input = prompt.check_relevance_prompt(test_data, i, s, use_prev_messages=args.message_history>0, demonstration=args.demonstration)
        response = api.generate(prompt_input)
        response = response.lower()

        if args.print_llm_response: 
            print('************** check_relevance_prompt **************')
            print(f"Prompt: {prompt_input}")
            print(f"Response: {response}")
            print('****************************************************')

        if 'yes' in response: return s

    return None


def get_pred(args, api, hypothesis, test_data, i, strategy, summary_high_reward, prompt):
    # get summary
    if strategy == 'best':
        summary = summary_high_reward
    elif strategy == 'relevant':
        summary = get_best_relevant_summary(args, api, hypothesis, test_data, i, prompt)
        print('Relevant summary:', summary)
        if summary == None:
            summary = summary_high_reward

    if args.with_reasoning:
        prompt_input = prompt.hypothesis_based_inference(summary, test_data, i, use_prev_messages=args.message_history>0, demonstration=args.demonstration)
    else:
        prompt_input = prompt.hypothesis_based_inference_without_reasoning(summary, test_data, i, use_prev_messages=args.message_history>0, demonstration=args.demonstration)
    response = api.generate(prompt_input)

    if args.print_llm_response: 
        print('******** hypothesis_based_inference (with/without reasoning) ********')
        print(f"Prompt: {prompt_input}")
        print(f"Response: {response}")
        print('*********************************************************************')

    pred = response.split("Answer")[1] if "Answer" in response else response
    pred = extract_label(args.task, pred)
    if args.task == 'diplomacy' and pred == 'other': pred = "False"

    return prompt_input, response, pred


def get_binary_vote_pred(args, api, hypothesis, test_data, i, prompt):
    answer_votes = {}
    for s, _ in hypothesis[:min(10, len(hypothesis))]: # check 10 hypotheses at most
        # check relevance
        prompt_input = prompt.check_relevance_prompt(test_data, i, s, use_prev_messages=(args.task=='diplomacy' and args.message_history>0), demonstration=args.demonstration)
        response = api.generate(prompt_input)
        response = response.lower()
        if 'yes' not in response: 
            # print('Not relevant')
            continue

        if args.print_llm_response: 
            print('******** check_relevance_prompt ********')
            print(f"Prompt: {prompt_input}")
            print(f"Response: {response}")
            print('****************************************')
        
        if args.with_reasoning:
            prompt_input = prompt.hypothesis_based_inference(s, test_data, i, use_prev_messages=args.message_history>0, demonstration=args.demonstration)
        else:
            prompt_input = prompt.hypothesis_based_inference_without_reasoning(s, test_data, i, use_prev_messages=args.message_history>0, demonstration=args.demonstration)
        response = api.generate(prompt_input)

        if args.print_llm_response: 
            print('******** hypothesis_based_inference (with/without reasoning) ********')
            print(f"Prompt: {prompt_input}")
            print(f"Response: {response}")
            print('*********************************************************************')

        if "Answer" in response:
            pred = response.split("Answer")[1]
        elif "Label" in response:
            pred = response.split("Label")[1]
        else:
            pred = response

        pred = extract_label(args.task, pred)

        if args.task == 'diplomacy' and pred == 'other': 
            pred = "False"
        
        answer_votes[pred] = answer_votes.get(pred, 0) + 1

    if answer_votes == {}:
        pred = random.choice(TASKS[args.task]().label_classes)
    else:
        pred = max(answer_votes, key=answer_votes.get)
    
    return test_data['label'][i], pred


def inference(args, hypothesis, test_data, api, prompt, strategy='best'):
    """
    Use the summary with the highest reward to make predictions

    Args:
        hypothesis: a dictionary of summaries and their rewards
        test_data: a list of (appearance, shoe_description) pairs
        api: the OpenAI API
        strategy: 'best' or 'relevant' or 'binary_vote'

    Returns:
        None
    """
    summary_high_reward, hypothesis = get_best_summary(hypothesis, args)
    results = []
    num_examples = get_num_examples(test_data)
    for i in range(num_examples):
        print('********** Example', i, '**********')
        label = test_data['label'][i]

        if strategy == 'binary_vote':
            label, pred = get_binary_vote_pred(args, api, hypothesis, test_data, i, prompt)
            print(f"Label: {label}")
            print(f"Prediction: {pred}")
            results.append({
                'label': label,
                'pred': pred
            })
        else:
            prompt_input, response, pred = get_pred(args, api, hypothesis, test_data, i, strategy, summary_high_reward, prompt)
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

    # compute metrics
    labels = [result['label'] for result in results]
    preds = [result['pred'] for result in results]
    if args.task in ['diplomacy', 'nli', 'diplomacy_swnames']:
        compute_binary_metrics(args.task, labels, preds)
    else:
        acc = sum([1 if labels[i] == preds[i] else 0 for i in range(len(labels))]) / len(labels)
        print(f'Accuracy: {acc}')

    return results


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--summary_pkl_path', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--strategy', type=str, default='best', choices=['best', 'relevant', 'binary_vote'])
    parser.add_argument('--num_test', type=int, default=-1)
    parser.add_argument('--message_history', type=int, default=0)
    parser.add_argument('--model', type=str, default='claude_2', choices=VALID_MODELS)
    parser.add_argument('--message', type=str, default='')
    parser.add_argument('--with_reasoning', type=str, default='no_', choices=['with_', 'no_'])
    parser.add_argument('--demonstration', type=str, default="no_", choices=["no_", "with_"], help='whether to use demonstration')
    parser.add_argument('--verbose', action='store_true', help='whether to print more information')
    parser.add_argument('--print_llm_response', action='store_true', help='whether to print the response from LLM')

    args = parser.parse_args()

    if args.with_reasoning == 'no_':
        args.with_reasoning = False
    else:
        args.with_reasoning = True
    assert type(args.with_reasoning) == bool

    if args.demonstration == 'no_':
        args.demonstration = False
    else:
        args.demonstration = True
    assert type(args.demonstration) == bool

    return args


def main():
    start_time = time.time()
    
    args = parse_args()

    # set up
    set_seed(args)
    test_data = get_test_data(args)
    api = LLMWrapper(args.model)
    prompt = PROMPT_DICT[args.task]()

    # load summaries formed from past interactions
    with open(args.summary_pkl_path, 'rb') as f:
        hypothesis = pickle.load(f)

    # inference: use the summary with the highest reward
    results = inference(args, hypothesis, test_data, api, prompt, strategy=args.strategy)
    prefix = args.summary_pkl_path.split('/')[-1].split('.')[0]
    folder = f'./outputs_{args.message}/{args.task}/test/'
    create_directory(folder)
    with open(f'{folder}/{prefix}_inference_test{args.num_test}_{args.strategy}_prev{args.message_history}_{args.model}_{"with" if args.with_reasoning else "no"}_reasoning.pkl', 'wb') as f:
        pickle.dump(results, f)

    # print time and estimated cost
    print(f'Time: {time.time() - start_time} seconds')
    if api.model in GPT_MODELS:
        print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == '__main__':
    main()