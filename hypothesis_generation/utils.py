from abc import ABC, abstractmethod
import pickle
import math
from typing import Dict, List
import torch
import re
import os
import numpy as np
import random
import openai
import vllm
import asyncio
import tqdm
from openai import AsyncOpenAI, OpenAI

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from pprint import pprint
from anthropic import Anthropic

from .LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from .consts.model_consts import (
    GPT_MODELS,
    CLAUDE_MODELS,
    LLAMA_MODELS,
    MISTRAL_MODELS,
    VALID_MODELS,
)

from .tasks import BaseTask

PORT = int(os.environ.get("PORT"))

POSITIVE_LABELS = {
    "hotel_reviews": "truthful",
    "headline_binary": "headline 1",
    "retweet": "first",
}


class LocalModelAPI:
    def __init__(self, localmodel, **kwargs):
        self.localmodel = localmodel

    def generate(self, messages, max_tokens=500, temperature=1e-5, **kwargs):
        if isinstance(self.localmodel, vllm.LLM):
            return self._vllm_generate([messages], max_tokens, temperature, **kwargs)[0]
        else:
            output = self._generate(messages, max_tokens, temperature, **kwargs)
            return output[0]["generated_text"][-1]["content"]

    def batched_generate(
        self, messages: List[Dict[str, str]], max_tokens=500, temperature=1e-5, **kwargs
    ):
        if isinstance(self.localmodel, vllm.LLM):
            return self._vllm_generate(messages, max_tokens, temperature, **kwargs)
        else:
            output = self._generate(messages, max_tokens, temperature, **kwargs)
            return [o[0]["generated_text"][-1]["content"] for o in output]

    def _generate(self, messages, max_tokens=500, temperature=1e-5, **kwargs):
        output = self.localmodel(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return output

    def _vllm_generate(
        self, messages: List[Dict[str, str]], max_tokens=500, temperature=1e-5, **kwargs
    ):

        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        tokenizer = self.localmodel.get_tokenizer()
        formatted_prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        output = self.localmodel.generate(formatted_prompts, sampling_params)
        return [o.outputs[0].text for o in output]


class LLMWrapper(ABC):
    def __init__(self, model, api, use_cache=1):
        self.model = model
        self.use_cache = use_cache
        self.api = api

    @classmethod
    def from_model(cls, model, use_cache=1, use_vllm=False, **kwargs):
        if model in GPT_MODELS.keys():
            return GPTWrapper(model, use_cache=use_cache, **kwargs)
        elif model in CLAUDE_MODELS.keys():
            return ClaudeWrapper(model, use_cache=use_cache, **kwargs)
        elif model in (LLAMA_MODELS + MISTRAL_MODELS):
            return LocalModelWrapper(
                model, use_cache=use_cache, use_vllm=use_vllm, **kwargs
            )
        else:
            raise NotImplementedError

    @abstractmethod
    def generate(self, messages, max_tokens=500, temperature=1e-5, **kwargs):
        pass


class GPTWrapper(LLMWrapper):
    def __init__(self, model, use_cache=1, **kwargs):
        super().__init__(
            model, use_cache=use_cache, api=self._setup(use_cache=use_cache, **kwargs)
        )

    def _setup(self, use_cache=1, max_retry=30, **kwargs):
        if use_cache == 1:
            return OpenAIAPICache(mode="chat", port=PORT, max_retry=max_retry)
        else:
            return OpenAI()

    def batched_generate(
        self,
        messages: List[Dict[str, str]],
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        n=1,
        **kwargs,
    ):
        client = AsyncOpenAI()
        status_bar = tqdm.tqdm(total=len(messages))
        # TODO: retry on failure
        async def _async_generate(sem, messages, **kwargs):
            async with sem:
                resp = await client.chat.completions.create(
                    model=GPT_MODELS[self.model],
                    messages=messages,
                    **kwargs,
                )
                status_bar.update(1)
                return resp

        sem = asyncio.Semaphore(max_concurrent)
        tasks = [
            _async_generate(
                sem,
                messages[i],
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                **kwargs,
            )
            for i in range(len(messages))
        ]
        loop = asyncio.get_event_loop()
        resp = loop.run_until_complete(asyncio.gather(*tasks))
        return [r.choices[0].message.content for r in resp]

    def generate(self, messages, max_tokens=500, temperature=1e-5, n=1, **kwargs):
        # Call OpenAI's API to generate inference

        if self.use_cache == 1:
            resp = self.api.generate(
                model=GPT_MODELS[self.model],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                messages=messages,
                **kwargs,
            )
        else:
            resp = self.api.chat.completions.create(
                model=GPT_MODELS[self.model],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                messages=messages,
                **kwargs,
            )

        return resp.choices[0].message.content


class ClaudeWrapper(LLMWrapper):
    def __init__(self, model, use_cache=1, **kwargs):
        super().__init__(
            model, use_cache=use_cache, api=self._setup(use_cache=use_cache, **kwargs)
        )

    def _setup(self, use_cache=1, max_retry=30, **kwargs):
        # TODO: get api key from environment variable
        api_key = open(f"./claude_key.txt", "r").read().strip()
        client = Anthropic(
            api_key=api_key,
        )

        api = ClaudeAPICache(client=client, port=PORT, max_retry=max_retry)

        if use_cache == 1:
            return api
        else:
            return client

    def generate(self, messages, max_tokens=500, temperature=1e-5, **kwargs):
        for idx, msg in enumerate(messages):
            if msg["role"] == "system":
                system_prompt = messages.pop(idx)["content"]

        if self.use_cache == 1:
            response = self.api.generate(
                model=CLAUDE_MODELS[self.model],
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,  # <-- system prompt
                messages=messages,  # <-- user prompt**kwargs
                **kwargs,
            )
        else:
            response = self.api.messages.create(
                model=CLAUDE_MODELS[self.model],
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,  # <-- system prompt
                messages=messages,  # <-- user prompt
                **kwargs,
            )
        if response == "Output blocked by content filtering policy":
            return None
        return response.content[0].text


class LocalModelWrapper(LLMWrapper):
    def __init__(self, model, use_cache=1, use_vllm=False, **kwargs):
        super().__init__(
            model,
            use_cache=use_cache,
            api=self._setup(model, use_cache=use_cache, use_vllm=use_vllm, **kwargs),
        )

    def _setup(
        self,
        model,
        cache_dir=f"./local_models_cache",
        path_name=None,
        use_cache=1,
        max_retry=30,
        use_vllm=False,
        **kwargs,
    ):
        if path_name is None:
            if model == "Mixtral-8x7B":
                path_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            elif model == "Mistral-7B":
                path_name = "mistralai/Mistral-7B-Instruct-v0.2"
            elif "Llama" in model:
                path_name = f"meta-llama/{model}-hf"
            else:
                raise ValueError(f"Model {model} not recognized.")

        if use_vllm:
            localmodel = vllm.LLM(
                model=path_name, tensor_parallel_size=torch.cuda.device_count()
            )
        else:
            localmodel = pipeline(
                "text-generation",
                device_map="auto",
                model=path_name,
                model_kwargs=kwargs,
            )

        client = LocalModelAPI(localmodel)
        api = LocalModelAPICache(client=client, port=PORT, max_retry=max_retry)

        if use_cache == 1:
            return api
        else:
            return client

    def generate(self, messages, max_tokens=500, temperature=1e-5, **kwargs):
        return self.api.generate(messages, max_tokens, temperature, **kwargs)

    def batched_generate(self, messages, max_tokens=500, temperature=1e-5, **kwargs):
        return self.api.batched_generate(messages, max_tokens, temperature, **kwargs)


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


def extract_hypotheses(num_hypothesis, text):
    # Get the hypotheses (numbered sentences)
    pattern = re.compile(r"\d+\.\s(.+?)(?=\d+\.\s|\Z)", re.DOTALL)
    hypotheses = pattern.findall(text)
    if len(hypotheses) != num_hypothesis:
        print(f"Expected {num_hypothesis} hypotheses, but got {len(hypotheses)}.")

    return hypotheses


def extract_label(task_name, pred):
    task = BaseTask(task_name)
    return task.extract_label(pred)


def set_seed(seed):
    print(f"Setting seed to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
