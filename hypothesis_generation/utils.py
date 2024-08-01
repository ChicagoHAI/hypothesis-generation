from abc import ABC, abstractmethod
import pickle
import math
import torch
import re
import os
import numpy as np
import random
import openai

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from pprint import pprint

# from openai_api_cache import OpenAIAPICache
from anthropic import Anthropic

from .LLM_cache import ClaudeAPICache, MixtralAPICache, OpenAIAPICache, LlamaAPICache
from .consts.model_consts import (
    INST_WRAPPER,
    GPT_MODELS,
    CLAUDE_MODELS,
    LLAMA_MODELS,
    MISTRAL_MODELS,
)

from .tasks import BaseTask

PORT = int(os.environ.get("PORT"))

VALID_MODELS = (
    list(GPT_MODELS.keys()) + list(CLAUDE_MODELS.keys()) + LLAMA_MODELS + MISTRAL_MODELS
)

POSITIVE_LABELS = {
    "hotel_reviews": "truthful",
    "headline_binary": "headline 1",
    "retweet": "first",
}


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def transform_sys_prompt(prompt, instruction_wrapper, inst_in_sys=True):
    system_prompt = prompt[0]
    user_prompt = prompt[1]

    if inst_in_sys:
        # If we put instructions in system prompt, wrap them by corresponding tokens for each model
        system_prompt = (
            f"{instruction_wrapper[0]}{system_prompt}{instruction_wrapper[1]}"
        )
    else:
        # Else, we just put the default system prompt and put instructions in user prompt, default wrapped by ### Instructions ###
        system_prompt = "You're a helpful assistant."
        user_prompt = (
            f"{INST_WRAPPER['default'][0]}{prompt[0]}{INST_WRAPPER['default'][1]}"
            + prompt[1]
        )

    return system_prompt, user_prompt


class LlamaAPI:
    def __init__(self, llama):
        self.llama = llama

    def generate(self, system_prompt, user_prompt, max_tokens=500):
        output = self.llama(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=max_tokens,
        )
        return output[0]["generated_text"][-1]["content"]


class MixtralAPI:
    def __init__(self, mixtral):
        self.mixtral = mixtral

    # def generate(self, prompt, max_tokens=500):
    #     model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
    #     generated_ids = self.mistral.generate(
    #         **model_inputs,
    #         max_new_tokens=max_tokens,
    #         pad_token_id=self.tokenizer.eos_token_id,
    #     )
    #     output_text = self.tokenizer.batch_decode(
    #         generated_ids, skip_special_tokens=True
    #     )[0]
    #     output_text = output_text[len(prompt) :]
    #     return output_text
    def generate(self, system_prompt, user_prompt, max_tokens=500):
        output = self.llama(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=max_tokens,
        )
        return output[0]["generated_text"][-1]["content"]


class LLMWrapper(ABC):
    def __init__(self, model, api, use_cache=1):
        self.model = model
        self.use_cache = use_cache
        self.api = api

    @classmethod
    def from_model(cls, model, use_cache=1, **kwargs):
        if model in GPT_MODELS.keys():
            return GPTWrapper(model, use_cache=use_cache, **kwargs)
        elif model in CLAUDE_MODELS.keys():
            return ClaudeWrapper(model, use_cache=use_cache, **kwargs)
        elif model in LLAMA_MODELS:
            return LlamaWrapper(model, use_cache=use_cache, **kwargs)
        elif model in MISTRAL_MODELS:
            return MixtralWrapper(model, use_cache=use_cache, **kwargs)
        else:
            raise NotImplementedError

    @abstractmethod
    def generate(self, prompt, inst_in_sys=True, max_tokens=500):
        pass


class GPTWrapper(LLMWrapper):
    def __init__(self, model, use_cache=1, **kwargs):
        super().__init__(
            model, use_cache=use_cache, api=self._setup(use_cache=use_cache, **kwargs)
        )

    def _setup(self, use_cache=1, **kwargs):
        api = OpenAIAPICache(mode="chat", port=PORT)

        return api

    def generate(self, prompt, inst_in_sys=True, max_tokens=500):
        # Call OpenAI's GPT-3.5 API to generate inference

        # basically adding ### Instructions ### around the instructions prompt
        instruction_wrapper = INST_WRAPPER["default"]
        system_prompt, user_prompt = transform_sys_prompt(
            prompt, instruction_wrapper, inst_in_sys
        )
        if self.use_cache == 1:
            resp = self.api.generate(
                # model="gpt-3.5-turbo-0613",
                model=GPT_MODELS[self.model],
                temperature=0.7,
                max_tokens=max_tokens,
                n=1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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
                    {"role": "user", "content": user_prompt},
                ],
            )

        return resp["choices"][0]["message"]["content"]


class ClaudeWrapper(LLMWrapper):
    def __init__(self, model, use_cache=1, **kwargs):
        super().__init__(
            model, use_cache=use_cache, api=self._setup(use_cache=use_cache, **kwargs)
        )

    def _setup(self, use_cache=1, **kwargs):
        # TODO: get api key from environment variable
        api_key = open(f"./claude_key.txt", "r").read().strip()
        client = Anthropic(
            api_key=api_key,
        )

        api = ClaudeAPICache(client=client, port=PORT)

        if use_cache == 1:
            return api
        else:
            return client

    def generate(self, prompt, inst_in_sys=True, max_tokens=500):
        # basically adding ### Instructions ### around the instructions prompt
        instruction_wrapper = INST_WRAPPER["default"]
        system_prompt, user_prompt = transform_sys_prompt(
            prompt, instruction_wrapper, inst_in_sys
        )

        if self.use_cache == 1:
            response = self.api.generate(
                model=CLAUDE_MODELS[self.model],
                max_tokens=max_tokens,
                temperature=0,
                system=system_prompt,  # <-- system prompt
                messages=[{"role": "user", "content": user_prompt}],  # <-- user prompt
            )
        else:
            response = self.api.messages.create(
                model=CLAUDE_MODELS[self.model],
                max_tokens=max_tokens,
                temperature=0,
                system=system_prompt,  # <-- system prompt
                messages=[{"role": "user", "content": user_prompt}],  # <-- user prompt
            )
        if response == "Output blocked by content filtering policy":
            return None
        return response.content[0].text


class MixtralWrapper(LLMWrapper):
    def __init__(self, model, use_cache=1, **kwargs):
        super().__init__(
            model,
            use_cache=use_cache,
            api=self._setup(model, use_cache=use_cache, **kwargs),
        )

    def _setup(
        self,
        model,
        cache_dir=f"./Mixtral_cache",
        path_name=None,
        use_cache=1,
        **kwargs,
    ):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if path_name is None:
            if model == "Mixtral-8x7B":
                path_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            elif model == "Mistral-7B":
                path_name = "mistralai/Mistral-7B-Instruct-v0.2"
            else:
                raise ValueError(f"Model {model} not recognized.")

        # model = AutoModelForCausalLM.from_pretrained(
        #     path_name,
        #     cache_dir=cache_dir,
        #     device_map="auto",
        # )
        # tokenizer = AutoTokenizer.from_pretrained(path_name, cache_dir=cache_dir)

        mixtral = pipeline(
            "text-generation",
            device_map="auto",
            model=path_name,
            model_kwargs=kwargs,
        )

        client = MixtralAPI(mixtral)
        api = MixtralAPICache(client=client, port=PORT)

        if use_cache == 1:
            return api
        else:
            return client

    def generate(self, prompt, inst_in_sys=True, max_tokens=500):
        system_prompt = prompt[0]
        user_prompt = prompt[1]

        output = self.api.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        return output


class LlamaWrapper(LLMWrapper):
    def __init__(self, model, use_cache=1, **kwargs):
        super().__init__(
            model, use_cache=use_cache, api=self._setup(use_cache=use_cache, **kwargs)
        )

    def _setup(
        self,
        cache_dir=f"./llama_cache",
        path_name=None,
        use_cache=1,
        **kwargs,
    ):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if path_name is None:
            path_name = f"meta-llama/{self.model}-hf"

        llama = pipeline(
            "text-generation",
            device_map="auto",
            model=path_name,
            model_kwargs=kwargs,
        )

        client = LlamaAPI(llama)
        api = LlamaAPICache(client=client, port=PORT)

        if use_cache == 1:
            return api
        else:
            return client

    def generate(self, prompt, inst_in_sys=True, max_tokens=500):
        system_prompt = prompt[0]
        user_prompt = prompt[1]

        output = self.api.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        return output


def get_results(task_name, pred_list, label_list):
    """
    Compute tp, tn, fp, fn for binary classification.
    Note that if predicted output is 'other', it is considered as negative.
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    positive_label = POSITIVE_LABELS[task_name]
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
    task = BaseTask(task_name)
    return task.extract_label(pred)


def set_seed(seed):
    print(f"Setting seed to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
