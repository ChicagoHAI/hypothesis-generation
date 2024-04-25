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
# from openai_api_cache import OpenAIAPICache
from anthropic import Anthropic

from claude_api_cache import ClaudeAPICache, MixtralAPICache, OpenAIAPICache
from consts.model_consts import INST_WRAPPER, GPT_MODELS, CLAUDE_MODELS, LLAMA_MODELS, MISTRAL_MODELS

code_repo_path = os.environ.get("CODE_REPO_PATH")
PORT = int(os.environ.get("PORT"))

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable CODE_REPO_PATH not set.")


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
                 use_cache=1,
                 **kwargs):
        assert model in VALID_MODELS, print('Invalid model name: ', model)
        self.model = model
        self.use_cache = use_cache
        if self.model in GPT_MODELS.keys():
            self.api = self._set_up_chatgpt(use_cache=use_cache,
                                            **kwargs)
        elif self.model in CLAUDE_MODELS.keys():
            self.api = self._set_up_claude_2(use_cache=use_cache,
                                             **kwargs)
        elif self.model in LLAMA_MODELS:
            self.api = self._set_up_llama(use_cache=use_cache,
                                          **kwargs)
        elif self.model in MISTRAL_MODELS:
            self.api = self._set_up_mistral(model,
                                            use_cache=use_cache,
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
                        use_cache=1,
                        **kwargs):

        openai.api_key_path = f'{code_repo_path}/openai_key.txt'
        api = OpenAIAPICache(mode="chat", port=PORT)

        return api


    def _set_up_claude_2(self, 
                         use_cache=1,
                         **kwargs):
        api_key = open(f'{code_repo_path}/claude_key.txt', 'r').read().strip()
        client = Anthropic(
            api_key=api_key,
        )
        
        api = ClaudeAPICache(
            client=client,
            port=PORT
        )
        
        if use_cache == 1:
            return api
        else:
            return client

    def _set_up_llama(self,
                      cache_dir=f"{code_repo_path}/llama_cache",
                      path_name=None,
                      **kwargs):
        if torch.cuda.is_available(): 
            device = "cuda" 
        else: 
            device = "cpu"

        if path_name is None:
            path_name = f'meta-llama/{self.model}-hf'

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
                        use_cache=1,
                        **kwargs):
        if torch.cuda.is_available(): 
            device = "cuda" 
        else: 
            device = "cpu"

        if path_name is None:
            if model=='Mixtral-8x7B':
                path_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
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

        if use_cache == 1:
            return api
        else:
            return client
        
    def _chatgpt_generate(self, prompt, inst_in_sys=True, max_tokens=500):
        # Call OpenAI's GPT-3.5 API to generate inference

        system_prompt, user_prompt = transform_sys_prompt(self.model,prompt,inst_in_sys)
        if self.use_cache:
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
        else:
            resp = openai.ChatCompletion.create(
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
    
    def _claude_2_message_generate(self, prompt,inst_in_sys=True, max_tokens=500):
        system_prompt, user_prompt = transform_sys_prompt(self.model,prompt,inst_in_sys)

        if self.use_cache:
            response = self.api.generate(
                model=CLAUDE_MODELS[self.model],
                max_tokens=max_tokens,
                temperature=0,
                system=system_prompt, # <-- system prompt
                messages=[
                    {"role": "user", "content": user_prompt} # <-- user prompt
                ]
            )
        else:
            response = self.messages.create(
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

def extract_label(task_name, pred):
    task = TASKS[task_name]()
    return task.extract_label(pred)

def set_seed(args):
    print(f"Setting seed to {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
