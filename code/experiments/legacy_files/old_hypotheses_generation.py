# inductive reasoning with intermediate summary formation
# generate summaries based on incoming examples

import time
import random
import pickle
import sys
import math
import argparse
import sys
import os

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")

sys.path.append(f'{code_repo_path}/code')
from utils import LLMWrapper, SummaryInformation, get_num_examples, extract_label, create_directory, VALID_MODELS, GPT_MODELS, set_seed
from prompt import PROMPT_DICT
from data_loader import get_train_data


def create_hypothesis(args, hypotheses, train_data, index, prompt, verbose=False):
    """
    Given a list of examples, generate a new summary and add it to the set of hypotheses.
    Try to generate a summary that is different from existing summaries.
    """
    # do not use previous messages to create summary for diplomacy because it would be too long
    summary = prompt.information_prompt(train_data, index)
    
    if summary in hypotheses:
        print('Summary already exists: ', summary)
    else:
        # print('index: ', index)
        hypotheses[summary] = SummaryInformation(index=index, is_example=True, value=None)
        # print(f'hypotheses[summary]: {hypotheses[summary]}')
        hypotheses['num_example_summaries'].value += 1

    if verbose:
        print('************** create_hypothesis **************')
        print('Response: ', summary)
        print('***********************************************')


def is_useful(args, api, train_data, summary, example_idx, prompt, verbose=False):
    # checks whether the summary can be used to infer the example
    prompt_input = prompt.check_usefulness_prompt(train_data, example_idx, summary, use_prev_messages=(args.task=='diplomacy' and args.message_history > 0), demonstration=args.demonstration)

    response = api.generate(prompt_input)
    response = response.lower()
    pred = response.split('answer')[-1]

    pred = extract_label(args.task, pred)

    label = train_data['label'][example_idx]

    if verbose:
        # print('************** Check entailment **************')
        print('****************** is_useful ******************')
        # print('prompt: ', prompt_input)
        print('response: ', response)
        # print('label: ', label)
        # print('pred: ', pred)
        # print('label == pred: ', label == pred)
        print('***********************************************')

    if label == pred:
        return True
    else:
        return False


def can_generate_summary(args, api, train_data, sample_examples_idx, summary, prompt, verbose=False):
    """
    Given a summary and a list of examples, check whether the summary can be updated using the examples
    """
    prompt_input = prompt.can_generate_summary_prompt(summary, train_data, sample_examples_idx, use_prev_messages=(args.task=='diplomacy' and args.message_history > 0), demonstration=args.demonstration)

    response = api.generate(prompt_input)
    response = response.lower()
    response = response.replace(' ', '')

    if verbose:
        print('****************** can_generate_summary ******************')
        print('response: ', response)
        print('**********************************************************')

    if 'yes' in response:
        return True
    elif 'no' in response:
        return False
    else:
        if verbose:
            # print('response: ', response)
            print('Invalid response: ', response)
        return False


def parse_summary(response, num_summaries):
    """
    parse the response from GPT-3.5 to get the generated summaries

    assume the response is in the format of:
    1. summary, 2. summary, ... 5. summary

    return: a list of summaries
    """
    summaries = []
    for i in range(num_summaries, 0, -1):
        summary = response.split(f'{i}. ')[1]
        response = response.split(f'{i}. ')[0]
        summaries.append(summary)

    return summaries


def get_best_summary(args, api, train_data, sample_examples_idx, summaries, prompt, verbose=False):
    """
    Given a list of summaries, return the best summary that is most useful for the given examples (i.e., can be used to infer the examples)    
    """
    summary_acc = {}
    for summary in summaries:
        correct = 0
        for example_idx in sample_examples_idx:
            useful = 1 if is_useful(args, api, train_data, summary, example_idx, prompt, verbose=verbose) else 0
            correct += useful
        summary_acc[summary] = correct / len(sample_examples_idx)
    
    best_summary = max(summary_acc, key=summary_acc.get)
    best_summary_acc = summary_acc[best_summary]
    return best_summary, best_summary_acc


def generate_summary(args, api, train_data, sample_examples_idx, prompt, verbose=False):
    """
    Given a list of examples, return the summary that is most useful for the given examples (i.e., can be used to infer the examples) among all possible summaries
    """
    num_summaries = 5

    prompt_input = prompt.generate_hypothesis(train_data, sample_examples_idx, num_summaries, use_prev_messages=(args.task=='diplomacy' and args.message_history > 0), demonstration=args.demonstration)

    response = api.generate(prompt_input)
    summaries = parse_summary(response, num_summaries)
    best_summary, best_summary_acc = get_best_summary(args, api, train_data, sample_examples_idx, summaries, prompt, verbose=verbose)

    if verbose:
        # print('************** Generate summary **************')
        print('****************** response ******************')
        # print('prompt: ', prompt_input)
        print('response: ', response)
        # print('best_summary: ', best_summary)
        print('**********************************************')
    
    return best_summary, best_summary_acc


def revise_summary(args, train_data, api, hypotheses, summary, prompt, verbose=False):
    """
    Takes a summary and revise it to make it more useful using examples that are not used to generate the summary
    
    params:
        train_data
        api: GPT-3.5 API
        hypotheses: current hypotheses
        summary: summary to revise
        verbose: whether to print more info
    
    returns: 
        revised summary (return None if summary cannot be or is not revised)
    """
    unused_indices = hypotheses[summary].get_unused_indices()

    if can_generate_summary(args, api, train_data, unused_indices, summary, prompt, verbose=verbose):
        new_summary, new_summary_acc = generate_summary(args, api, train_data, unused_indices, prompt, verbose=verbose)
        if new_summary not in hypotheses: return new_summary, new_summary_acc
        print('Summary already exists: ', new_summary)

    return None, None


def update_summary(args, train_data, index, api, hypotheses, old_summary, prompt, verbose=False):
    """ Revise and add new summary."""
    # revise summary
    old_summary_info = hypotheses[old_summary]
    new_summary, new_summary_acc = revise_summary(args, train_data, api, hypotheses, old_summary, prompt, verbose=args.verbose)

    # we only add the new summary if it is different from the old summary
    if new_summary is not None and new_summary != old_summary:
        # add new summary
        hypotheses[new_summary] = SummaryInformation(index=index, is_example=False, value=None)

        # different ways to inherit the old summary's information
        # note that the reward is computed using acc and num_visits
        if args.inheritance == "reward":
            # inherit the reward of the old summary
            hypotheses[new_summary].acc = old_summary_info.acc
            hypotheses[new_summary].num_visits = old_summary_info.num_visits

            # note that we do not need to update reward here because we will update reward for all summaries 
            # after we have used the current example to visit top k summaries and generate new summaries
            print('*** inherit reward ***')
            print('old_summary_info.reward: ', old_summary_info.reward)
            print('old_summary_info.acc: ', old_summary_info.acc)
            print('old_summary_info.num_visits: ', old_summary_info.num_visits)
            print('')
            print('new_summary_info.reward: ', hypotheses[new_summary].reward)
            print('new_summary_info.acc: ', hypotheses[new_summary].acc)
            print('new_summary_info.num_visits: ', hypotheses[new_summary].num_visits)
            print('*** end inherit reward ***')

        elif args.inheritance == "acc":
            # inherit the accuracy of the old summary
            # set num_visits to args.revise_freq, because we used these number of examples to generate the new summary
            hypotheses[new_summary].acc = old_summary_info.acc
            hypotheses[new_summary].num_visits = args.revise_freq
            assert old_summary_info.num_unused == args.revise_freq

            # note that we do not need to update reward here because we will update reward for all summaries 
            # after we have used the current example to visit top k summaries and generate new summaries
            print('*** inherit acc ***')
            print('old_summary_info.reward: ', old_summary_info.reward)
            print('old_summary_info.acc: ', old_summary_info.acc)
            print('old_summary_info.num_visits: ', old_summary_info.num_visits)
            print('')
            print('new_summary_info.reward: ', hypotheses[new_summary].reward)
            print('new_summary_info.acc: ', hypotheses[new_summary].acc)
            print('new_summary_info.num_visits: ', hypotheses[new_summary].num_visits)
            print('*** end inherit reward ***')

        elif args.inheritance == "nothing":
            # do not inherit anything from the old summary
            hypotheses[new_summary].acc = new_summary_acc
            hypotheses[new_summary].num_visits = args.revise_freq

            # note that we do not need to update reward here because we will update reward for all summaries 
            # after we have used the current example to visit top k summaries and generate new summaries
            print('*** inherit nothing ***')
            print('old_summary_info.reward: ', old_summary_info.reward)
            print('old_summary_info.acc: ', old_summary_info.acc)
            print('old_summary_info.num_visits: ', old_summary_info.num_visits)
            print('')
            print('new_summary_info.reward: ', hypotheses[new_summary].reward)
            print('new_summary_info.acc: ', hypotheses[new_summary].acc)
            print('new_summary_info.num_visits: ', hypotheses[new_summary].num_visits)
            print('*** end inherit reward ***')

        else:
            raise NotImplementedError

        # print(f'hypotheses[summary]: {hypotheses[new_summary]}')
        assert len(hypotheses[new_summary].get_unused_indices()) == 0
        assert hypotheses[new_summary].num_unused == 0
        
    # reset unused indices for the old summary to be empty
    old_summary_info.update_info_after_revision_attempt(index)

def use_top_k_summaries(api, hypotheses, train_data, i, args, prompt):
    """
    Given a new example, we only use top k summaries to check usefulness 
    and update information including accuracy, count, and indices.
    Note that reward is updated in another functionbecause we need to update reward for all summaries.

    Returns: 
        - has_consistent_summary: whether the example is consistent with any of the top k summaries
    """
    has_consistent_summary = False

    # find top k summaries (rank by reward)
    top_k = sorted(hypotheses.items(), key=lambda x: x[1].reward, reverse=True)[:args.k]
    top_k = [x[0] for x in top_k]
    
    for summary in top_k:
        if summary == 'num_example_summaries': continue

        # use GPT-3.5 to check "entailment"
        useful = is_useful(args, api, train_data, summary, i, prompt, verbose=args.verbose)

        summary_info = hypotheses[summary]

        # update summary info
        if useful:
            # print('*** use_top_k_summaries ***')
            # print('before update info, summary_info.num_unused: ', summary_info.num_unused)
            summary_info.update_info_if_useful(i)
            # print('after update info, summary_info.num_unused: ', summary_info.num_unused)
            has_consistent_summary = True
            assert summary_info.num_unused <= args.revise_freq
            if summary_info.num_unused == args.revise_freq:
                # update summary: revise and add new summary, delete old summary
                update_summary(args, train_data, i, api, hypotheses, summary, prompt, verbose=args.verbose)
                if summary in hypotheses:
                    summary_info = hypotheses[summary]
                    assert summary_info.num_unused == 0
            # print('*** end use_top_k_summaries ***')
        else:
            summary_info.update_info_if_not_useful(i)
    
    return has_consistent_summary


def update_reward(hypotheses, num_examples, c):
    """
    update reward based on accuracy and num_visits with UCB algorithm
    """
    for summary in hypotheses:
        hypotheses[summary].update_reward(c, num_examples)


def add_example_to_random_summary(args, api, hypotheses, train_data, i, prompt, verbose=False):
    # randomly add the example to one of the existing summaries
    random_summary = random.choice(list(hypotheses.keys()))
    while random_summary == 'num_example_summaries':
        random_summary = random.choice(list(hypotheses.keys()))
    summary_info = hypotheses[random_summary]

    # update summary info
    # print('*** add_example_to_random_summary ***')
    summary_info.update_info_if_useful(i)

    # we may need to revise the summary after adding the new example
    assert summary_info.num_unused <= args.revise_freq
    if summary_info.num_unused == args.revise_freq:
        # update summary: revise and add new summary, delete old summary
        update_summary(args, train_data, i, api, hypotheses, random_summary, prompt, verbose=args.verbose)
        if random_summary in hypotheses:
            summary_info = hypotheses[random_summary]
            assert summary_info.num_unused == 0


def build_hypotheses(args, api, train_data, prompt):
    """
    Build a set of hypotheses based on the training data
    key: hypotheses
    value: [acc, num_visit, reward, # used to compute and save reward
            num_unused, [unused supporting example indices]]] # used for updating summaries
    """
    hypotheses = {
        'num_example_summaries': SummaryInformation(index=None, is_example=None, value=0)
    }
    num_train_examples = get_num_examples(train_data)
    for i in range(num_train_examples):

        print('************* Instance', i, '*************')
        print('hypotheses:', len(hypotheses))

        # only use top k summaries to check entailment and update info
        has_consistent_summary = use_top_k_summaries(api, hypotheses, train_data, i, args, prompt)

        if has_consistent_summary != True:
            # if no summary is consistent with the example
            if hypotheses['num_example_summaries'].value < args.m:
                # we save the example for later
                create_hypothesis(args, hypotheses, train_data, i, prompt, verbose=args.verbose) 
            else:
                # or randomly add to an existing summary
                add_example_to_random_summary(args, api, hypotheses, train_data, i, prompt, verbose=args.verbose)
        
        # update rewards
        # (we always have actual rewards after we have used the current example to visit top k summaries and generate new summaries)
        update_reward(hypotheses, i+1, args.c)

        # save hypotheses
        if (i+1) % args.save_every == 0:
            print('Saving hypotheses...')
            print('Step:', i+1)
            
            file_path = f'{args.output_path_prefix}_step{i+1}.pkl'
            save_hypotheses(hypotheses, file_path)
    
    file_path = f'{args.output_path_prefix}_FINAL_step{num_train_examples}.pkl'
    save_hypotheses(hypotheses, file_path)


def save_hypotheses(hypotheses, file_path):
    # save the summaries
    directory = os.path.dirname(file_path)
    create_directory(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(hypotheses, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_train', type=int, default=25, help='Number of training examples.')
    parser.add_argument('--k', type=int, default=-1, help='Number of summaries to check for each new example.')
    parser.add_argument('--c', type=float, default=5e-1, help='Exploration parameter. Note that I use alpha in the paper.')
    parser.add_argument('--m', type=int, default=10, help='Maximum number of single-instance summaries to keep. Note that I use mu in the paper.')
    parser.add_argument('--revise_freq', type=int, default=4, help='If there are enough examples to revise a summary, we revise it. Note that I use c in the paper.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--verbose', action='store_true', help='Print the response from LLM.')
    parser.add_argument('--task', type=str, choices=['shoe', 
                                                     'diplomacy', 
                                                     'nli', 
                                                     'diplomacy_swnames', 
                                                     'sst',
                                                     'binary_sst',
                                                     'original_sst',
                                                     'binary_original_sst',
                                                     'hotel_reviews',
                                                     'headline_binary'
                                                     ], help='task to run (shoe or diplomacy)')
    parser.add_argument('--message_history', type=int, default=0, help='Number of previous messages to use as context. This only applies to the Diplomacy dataset.')
    parser.add_argument('--model', type=str, default='chatgpt', choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--message', type=str, default='no_message', help='A note on the experiment setting.')
    parser.add_argument('--inheritance', type=str, default='nothing', choices=['reward', 'acc', 'nothing'], help='Describes what information to to inherit from the old summary.')
    parser.add_argument('--demonstration', type=str, default="no_", choices=["no_", "with_"], help='whether to use demonstration')
    parser.add_argument('--output_path_prefix', type=str, default=None, help='Path to save the hypotheses.')
    parser.add_argument('--save_every', type=int, default=100, help='Save hypotheses every k steps.')

    args = parser.parse_args()

    if args.demonstration == 'no_':
        args.demonstration = False
    else:
        args.demonstration = True
    assert type(args.demonstration) == bool

    return args


def main():
    start_time = time.time()

    args = parse_args()
    set_seed(args)
    train_data = get_train_data(args)
    api = LLMWrapper(args.model)
    prompt = PROMPT_DICT[args.task]()

    build_hypotheses(args, api, train_data, prompt)

    print(f'Time: {time.time() - start_time} seconds')
    if api.model in GPT_MODELS:
        print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == '__main__':
    main()