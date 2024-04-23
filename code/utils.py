import pickle
import math
import torch
import re
import os
import numpy as np
import random
import openai

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoModelForCausalLM, AutoTokenizer
from pprint import pprint
from tasks import TASKS
from openai_api_cache import OpenAIAPICache
from anthropic import Anthropic

from claude_api_cache import ClaudeAPICache, MixtralAPICache
from consts.model_consts import INST_WRAPPER

code_repo_path = os.environ.get("CODE_REPO_PATH")
server = os.environ.get("SERVER")
PORT = int(os.environ.get("PORT"))

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable CODE_REPO_PATH not set.")

if server:
    print(f"Server: {server}")
else:
    print("Environment variable SERVER not set.")


GPT_MODELS = {
    'turbo35_0613': 'gpt-3.5-turbo-0613',
    'turbo35_1106': 'gpt-3.5-turbo-1106',
    'turbo4': 'gpt-4-1106-preview',
}

CLAUDE_MODELS = {
    "claude_2": "claude-2.1",
    "claude-2.1": "claude-2.1",
    "claude-2.0": "claude-2.0",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-opus": "claude-3-opus-20240229"
}

LLAMA_MODELS = [
    'Llama-2-7b',
    'Llama-2-7b-chat',
    'Llama-2-13b',
    'Llama-2-13b-chat',
    'Llama-2-70b',
    'Llama-2-70b-chat',
]

MISTRAL_MODELS = [
    'Mixtral-8x7B',
    'Mistral-7B'
]

VALID_MODELS = list(GPT_MODELS.keys()) + list(CLAUDE_MODELS.keys()) + LLAMA_MODELS + MISTRAL_MODELS


POSITIVE_LABELS = {
    'hotel_reviews': 'truthful',
    'headline_binary': 'headline 1',
    'retweet': 'first',
}

def get_device():
    if torch.cuda.is_available(): 
        device = "cuda" 
    else: 
        device = "cpu" 
    return device


def transform_sys_prompt(model,prompt,inst_in_sys=True):
    system_prompt = prompt[0]
    user_prompt = prompt[1]

    if model in MISTRAL_MODELS:
        # Mistral always requires putting instructions around the [INST] [/INST] tokens
        instruction_wrapper = INST_WRAPPER['mistral']
        system_prompt = f'{instruction_wrapper[0]}{system_prompt}{instruction_wrapper[1]}'
        return system_prompt, user_prompt
    elif model in LLAMA_MODELS:
        # For llama, because the system tokens are different, we handle it in _llama_generate
        instruction_wrapper = INST_WRAPPER['llama']
    else:
        # Default model here means for GPT and Claude, this is basically adding ### Instructions ### around the instructions prompt
        instruction_wrapper = INST_WRAPPER['default']

    if inst_in_sys:
        # If we put instructions in system prompt, wrap them by corresponding tokens for each model
        system_prompt = f'{instruction_wrapper[0]}{system_prompt}{instruction_wrapper[1]}'
    else:
        # Else, we just put the default system prompt and put instructions in user prompt, default wrapped by ### Instructions ###
        system_prompt = "You're a helpful assistant."
        user_prompt = f"{INST_WRAPPER['default'][0]}{prompt[0]}{INST_WRAPPER['default'][1]}"+prompt[1]
        
    return system_prompt, user_prompt


class LlamaAPI:
    def __init__(self, llama, tokenizer, device):
        self.llama = llama
        self.tokenizer = tokenizer
        self.device = device


    def generate(self, prompt, max_tokens=500):
        inputs = self.tokenizer(text=prompt, return_tensors="pt").to(self.device)
        out = self.llama.generate(inputs=inputs.input_ids, max_new_tokens=max_tokens)
        output_text = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        # remove the prompt from the output
        output_text = output_text[len(prompt):]
        return output_text


class MistralAPI:
    def __init__(self, mistral, tokenizer, device):
        self.mistral = mistral
        self.tokenizer = tokenizer
        self.device = device


    def generate(self, prompt, max_tokens=500):
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self.mistral.generate(**model_inputs, 
                                              max_new_tokens=max_tokens, 
                                              pad_token_id=self.tokenizer.eos_token_id)
        output_text = self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]
        output_text = output_text[len(prompt):]
        return output_text


class LLMWrapper:
    def __init__(self, 
                 model,
                 **kwargs):
        assert model in VALID_MODELS, print('Invalid model name: ', model)
        self.model = model
            
        if self.model in GPT_MODELS.keys():
            self.api = self._set_up_chatgpt(**kwargs)
        elif self.model in CLAUDE_MODELS.keys():
            self.api = self._set_up_claude_2(**kwargs)
        elif self.model in LLAMA_MODELS:
            self.api = self._set_up_llama(**kwargs)
        elif self.model in MISTRAL_MODELS:
            self.api = self._set_up_mistral(model,
                                            **kwargs)
        else:
            raise NotImplementedError

    def generate(self, prompt, inst_in_sys=True, max_tokens=500):
        if self.model in GPT_MODELS.keys():
            return self._chatgpt_generate(prompt, inst_in_sys, max_tokens)
        elif self.model in CLAUDE_MODELS.keys():
            return self._claude_2_message_generate(prompt, inst_in_sys, max_tokens)
        elif self.model in LLAMA_MODELS:
            return self._llama_generate(prompt, inst_in_sys, max_tokens)
        elif self.model in MISTRAL_MODELS:
            return self._mistral_generate(prompt, inst_in_sys, max_tokens)
        else:
            raise NotImplementedError
        

    def _set_up_chatgpt(self,
                        **kwargs):

        openai.api_key_path = f'{code_repo_path}/openai_key.txt'
        api = OpenAIAPICache(mode="chat", port=PORT)

        return api


    def _set_up_claude_2(self, **kwargs):
        api_key = open(f'{code_repo_path}/claude_key.txt', 'r').read().strip()
        client = Anthropic(
            api_key=api_key,
        )
        
        api = ClaudeAPICache(
            client=client,
            port=PORT
        )

        return api

    def _set_up_llama(self,
                      cache_dir=f"{code_repo_path}/llama_cache",
                      path_name=None,
                      **kwargs):
        if torch.cuda.is_available(): 
            device = "cuda" 
        else: 
            device = "cpu"

        if path_name is None:
            if server == 'indigo':
                path_name="/data/LLAMA2_chat_hf/llama2_chat_7B/"
            elif server == 'dsi':
                path_name="/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-7b-chat-hf/"
            else:
                raise ValueError(f"Server {server} not recognized.")

        config = LlamaConfig.from_pretrained(path_name, cache_dir=cache_dir)
        tokenizer = LlamaTokenizer.from_pretrained(path_name, cache_dir=cache_dir)
        llama = LlamaForCausalLM.from_pretrained(path_name, 
                                                 config=config, 
                                                 cache_dir=cache_dir,
                                                 device_map='auto',
        )


        api = LlamaAPI(llama, tokenizer, device)

        return api
    
    def _set_up_mistral(self,
                        model,
                        cache_dir=f"{code_repo_path}/Mixtral_cache",
                        path_name=None,
                        **kwargs):
        if torch.cuda.is_available(): 
            device = "cuda" 
        else: 
            device = "cpu"

        if path_name is None:
            if model=='Mixtral-8x7B':
                path_name = "/net/projects/chai-lab/tejes/Mixtral-8x7B-Instruct-v0.1"
            elif model == 'Mistral-7B':
                path_name = "mistralai/Mistral-7B-Instruct-v0.2"
            else:
                raise ValueError(f"Model {model} not recognized.")

        model = AutoModelForCausalLM.from_pretrained(path_name,
                                                     cache_dir=cache_dir,
                                                     device_map='auto',
                                                     )
        tokenizer = AutoTokenizer.from_pretrained(path_name,
                                                  cache_dir=cache_dir)
        
        client = MistralAPI(model,tokenizer,device)
        api = MixtralAPICache(
            client=client,
            port=PORT
        )

        # return MistralAPI(model,tokenizer,device)
        return api
        
    def _chatgpt_generate(self, prompt, inst_in_sys=True, max_tokens=500):
        # Call OpenAI's GPT-3.5 API to generate inference

        system_prompt, user_prompt = transform_sys_prompt(self.model,prompt,inst_in_sys)
        resp = self.api.generate(
            # model="gpt-3.5-turbo-0613",
            model=GPT_MODELS[self.model],
            temperature=0.7,
            max_tokens=max_tokens,
            n=1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return resp['choices'][0]['message']['content']
    
    # def _claude_2_generate(self, prompt, system_prompt="You are a helpful assistant."):
    #     completion = self.api.completions.create(
    #         model="claude-2.1",
    #         max_tokens_to_sample=1000,
    #         temperature=0,
    #         prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
    #     )
    #     return completion.completion

    def _claude_2_message_generate(self, prompt,inst_in_sys=True, max_tokens=500):
        system_prompt, user_prompt = transform_sys_prompt(self.model,prompt,inst_in_sys)

        response = self.api.generate(
            model=CLAUDE_MODELS[self.model],
            max_tokens=max_tokens,
            temperature=0,
            system=system_prompt, # <-- system prompt
            messages=[
                {"role": "user", "content": user_prompt} # <-- user prompt
            ]
        )

        if response == "Output blocked by content filtering policy":
            return None
        return response.content[0].text

    def _mistral_generate(self, prompt,inst_in_sys=True, max_tokens=500):
        system_prompt, user_prompt = transform_sys_prompt(self.model,prompt,inst_in_sys)
        output = self.api.generate(
            prompt=system_prompt+user_prompt,
            max_tokens=max_tokens)
        return output
    
    def _llama_generate(self, prompt,inst_in_sys=True, max_tokens=500):
        system_prompt, user_prompt = transform_sys_prompt(self.model,prompt,inst_in_sys)
        prompt = f"""[INST] <<SYS>> {system_prompt} <</SYS>>""" + user_prompt + " [/INST]"
        output = self.api.generate(prompt, max_tokens)
        return output


def get_dialogue_id(num_messages, message_id):
    # get dialogue id based on message idx
    # num_messages: list of number of messages in each dialogue
    # message_id: index of the message
    # return: dialogue id
    for i in range(len(num_messages)):
        if message_id < num_messages[i]:
            return i
        else:
            message_id -= num_messages[i]


def get_results(args, pred_list, label_list):
    """
    Compute tp, tn, fp, fn for binary classification.
    Note that if predicted output is 'other', it is considered as negative.
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    positive_label = POSITIVE_LABELS[args.task]
    for i in range(len(pred_list)):
        pred = pred_list[i]
        label = label_list[i]
        if pred == positive_label and label == positive_label:
            tp += 1
        elif pred == positive_label and label != positive_label:
            fp += 1
        elif pred != positive_label and label == positive_label:
            fn += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def get_num_examples(data):
    key = list(data.keys())[0]
    return len(data[key])


def extract_label(task_name, pred):
    task = TASKS[task_name]()
    return task.extract_label(pred)


def print_info(path):
    """
    Reads the hypothesis pickle file at path,\
    print the top 10 summaries with the highest rewards.
    """
    hypothesis = pickle.load(open(path, 'rb'))

    print('len(hypothesis)', len(hypothesis))

    # sort summaries by reward (third element in value)
    sorted_hyp = sorted(hypothesis.items(), key=lambda x: x[1][2], reverse=True)
    pprint([s[0] for s in sorted_hyp[:10] if len(s[0])<300])

    # print their rewards
    # pprint([s[1][2] for s in sorted_hyp[:10] if len(s[0])<300])


def get_hypothesis_freq(print_log_path):
    """
    For analyzing shoe recommendation results.

    Given a print_log_path (txt file), get the frequency of each hypothesis.
    """
    hypothesis_info = {} # key: hypothesis, value: (count, correct count)

    print_log = open(print_log_path, 'r').readlines()
    print_log = [l.strip() for l in print_log]

    # the selected hypothesis start with "Relevant summary: " in the line
    selected_hyp = [l[18:] for l in print_log if l.startswith("Relevant summary: ")]
    # selected_hyp = [l[13:] for l in print_log if l.startswith("We know that ") and l[13:] != '']
    labels = [l[7:] for l in print_log if l.startswith("Label: ")]
    preds = [l[12:] for l in print_log if l.startswith("Prediction: ")]

    for i in range(len(selected_hyp)):
        hyp = selected_hyp[i]
        label = labels[i]
        pred = preds[i]
        correctness = 1 if label == pred else 0
        if hyp in hypothesis_info:
            hypothesis_info[hyp][0] += 1
            hypothesis_info[hyp][1] += correctness
        else:
            hypothesis_info[hyp] = [1, correctness]

    # print hypothesis sorted by frequency (descending)
    sorted_hyp = sorted(hypothesis_info.items(), key=lambda x: x[1], reverse=True)
    pprint(sorted_hyp)


def convert_appearance_to_features(description):
    """
    Convert appearance descriptions to features (words)

    For example:
    parse the sentence 'a young and short man with red hat, green shirt, and a small black bag'
    into ['young', 'short', 'man', 'red', 'green', 'small', 'black']
    """
    # get rid of the punctuations
    description = description.replace(',', '')

    # split the sentence into words
    words = description.split()
    
    # keep the desired words
    keep_idx = [1, 3, 4, 6, 8, 12, 13]
    features = [words[i] for i in keep_idx]

    return features


def get_feature_vectors(word_features):
    """
    Convert features to feature vectors

    Given a list of list of words, we want to convert them into vectors. 
    """
    feature1_indices = {'young': 0, 'old': 1}
    feature2_indices = {'short': 0, 'tall': 1}
    feature3_indices = {'man': 0, 'woman': 1}
    feature4_indices = {'white': 0, 'black': 1, 'green': 2, 'blue': 3, 'red': 4, 'orange': 5}
    feature5_indices = {'white': 0, 'black': 1, 'green': 2, 'blue': 3, 'red': 4, 'orange': 5}
    feature6_indices = {'small': 0, 'large': 1}
    feature7_indices = {'white': 0, 'black': 1, 'green': 2, 'blue': 3, 'red': 4, 'orange': 5}

    feature_dicts = [feature1_indices, feature2_indices, feature3_indices, feature4_indices, feature5_indices, feature6_indices, feature7_indices]

    feature_vectors = []
    for feature in word_features:
        feature_vector = np.zeros(7)

        for i in range(7):
            feature_vector[i] = feature_dicts[i].get(feature[i], -1)
            if feature_vector[i] == -1:
                print(f'feature {i} not found, it is ', feature[i])

        feature_vectors.append(feature_vector)

    # turn to np array
    feature_vectors = np.array(feature_vectors)
    
    return feature_vectors


def get_shoe_labels(labels_text):
    """
    Given a list of labels in text format, convert them to numerical format.
    """
    label_indices = {'white': 0, 'black': 1, 'green': 2, 'blue': 3, 'red': 4, 'orange': 5}
    labels = [label_indices[label] for label in labels_text]
    labels = np.array(labels)
    return labels


def prep_shoe_data(shoe_data):
    """
    input: 
        shoe_data: dict, keys: 'appearance', 'label'

    output:
        x: numpy array of shape (num_examples, num_features)
        y: numpy array of shape (num_examples, )
    """
    features = [convert_appearance_to_features(appearance) for appearance in shoe_data['appearance']]
    x = get_feature_vectors(features)
    y = get_shoe_labels(shoe_data['label'])

    assert len(x) == len(y)

    return x, y


def get_f1(directory, seed, inference_model):
    file_name = f"{directory}/seed{seed}_{inference_model}.txt"
    # macro f1 is the line starting with "Macro F1: "
    lines = open(file_name, 'r').readlines()
    f1_line = [l for l in lines if l.startswith("Macro F1: ")][0]
    f1 = float(f1_line[len("Macro F1: "):])
    return f1


def get_average_f1(directory, seeds, inference_model):
    """
    Given a directory containing results from multiple seeds, compute the average f1 score.
    """
    f1s = []
    for seed in seeds:
        f1 = get_f1(directory, seed, inference_model)
        f1s.append(f1)
    
    avg_f1 = sum(f1s) / len(f1s)
    return avg_f1


def get_ours_acc(prefix, seed, inference_model, step, llm):
    file_name = f"{prefix}_seed{seed}_{inference_model}_{llm}_step{step}.txt"

    # accuracy is the line starting with "Accuracy: "
    lines = open(file_name, 'r').readlines()
    acc_line = [l for l in lines if l.startswith("Accuracy: ")][0]
    acc = float(acc_line[len("Accuracy: "):])
    return acc


def create_directory(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def extract_hypotheses(args, text):
    pattern = re.compile(r"Response:(.+?)response length", re.DOTALL)
    response_match = pattern.search(text)
    if response_match:
        # Extract the text between "Response" and "response length"
        extracted_text = response_match.group(1).strip()
        print("Extracted Text:", extracted_text)
    else:
        print("No match found.")
        return []
    # Get the hypotheses (numbered sentences)
    pattern = re.compile(r"\d+\.\s(.+?)(?=\d+\.\s|\Z)", re.DOTALL)
    hypotheses = pattern.findall(extracted_text)
    if len(hypotheses) != args.num_hypothesis:
        print(f"Expected {args.num_hypothesis} hypotheses, but got {len(hypotheses)}.")

    return hypotheses


def set_seed(args):
    print(f"Setting seed to {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


def reverse_order(args, data):
    if args.task not in ['retweet','headline_binary']:
        raise ValueError(f'{args.task} cannot be reversed.')
    elif args.task == 'headline_binary':
        labels = ['headline 1', 'headline 2']
        text_name = 'headline'
    else:
        labels = ['first', 'second']
        text_name = 'tweets'

    reversed_text = []
    reversed_label = []
    for i in range(len(data['label'])):
        reversed_text.append([data[text_name][i][1],data[text_name][i][0]])
        if data['label'][i] == labels[0]:
            reversed_label.append(labels[1])
        else:
            reversed_label.append(labels[0])
    reversed_data = {text_name: reversed_text,
                     'label': reversed_label}
    return reversed_data

