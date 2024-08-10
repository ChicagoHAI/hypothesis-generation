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
from anthropic import AsyncAnthropic, Anthropic

from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from pprint import pprint

from .LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from .tasks import BaseTask

PORT = int(os.environ.get("PORT"))


class LLMWrapper(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate(
        self, messages, use_cache=1, max_tokens=500, temperature=1e-5, **kwargs
    ):
        pass

    @abstractmethod
    def batched_generate(
        self, messages, use_cache=1, max_tokens=500, temperature=1e-5, **kwargs
    ):
        pass


class GPTWrapper(LLMWrapper):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self._setup(**kwargs)

    def _setup(self, max_retry=30, **kwargs):
        self.api = OpenAI()
        self.api_with_cache = OpenAIAPICache(port=PORT, max_retry=max_retry, **kwargs)
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        n=1,
        **kwargs,
    ):
        if len(messages) == 0:
            return []

        client = AsyncOpenAI()
        status_bar = tqdm.tqdm(total=len(messages))

        # TODO: retry on failure
        async def _async_generate(sem, messages, **kwargs):
            async with sem:
                resp = await client.chat.completions.create(
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

    def batched_generate(
        self, messages, use_cache=1, max_tokens=500, temperature=1e-5, n=1, **kwargs
    ):
        if use_cache == 1:
            return self.api_with_cache.batched_generate(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                **kwargs,
            )
        return self._batched_generate(messages, max_tokens, temperature, n, **kwargs)

    def _generate(self, messages, max_tokens=500, temperature=1e-5, n=1, **kwargs):
        resp = self.api.chat.completions.create(
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content

    def generate(
        self, messages, use_cache=1, max_tokens=500, temperature=1e-5, n=1, **kwargs
    ):
        # Call OpenAI's API to generate inference

        if use_cache == 1:
            resp = self.api_with_cache.generate(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                messages=messages,
                **kwargs,
            )
        else:
            resp = self._generate(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                messages=messages,
                **kwargs,
            )

        return resp


class ClaudeWrapper(LLMWrapper):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self._setup(**kwargs)

    def _setup(self, max_retry=30, **kwargs):
        self.api = Anthropic()
        self.api_with_cache = ClaudeAPICache(
            client=self.api, port=PORT, max_retry=max_retry, **kwargs
        )
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        if len(messages) == 0:
            return []

        client = AsyncAnthropic()
        status_bar = tqdm.tqdm(total=len(messages))

        # TODO: retry on failure
        async def _async_generate(sem, messages, **kwargs):
            for idx, msg in enumerate(messages):
                if msg["role"] == "system":
                    system_prompt = messages.pop(idx)["content"]
                    break

            async with sem:
                resp = await client.messages.create(
                    system=system_prompt,
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
                **kwargs,
            )
            for i in range(len(messages))
        ]
        loop = asyncio.get_event_loop()
        resp = loop.run_until_complete(asyncio.gather(*tasks))
        return [r.content[0].text for r in resp]

    def batched_generate(
        self, messages, use_cache=1, max_tokens=500, temperature=1e-5, **kwargs
    ):
        if use_cache == 1:
            return self.api_with_cache.batched_generate(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        return self._batched_generate(messages, max_tokens, temperature, **kwargs)

    def _generate(self, messages, max_tokens=500, temperature=1e-5, **kwargs):
        for idx, msg in enumerate(messages):
            if msg["role"] == "system":
                system_prompt = messages.pop(idx)["content"]
                break

        response = self.api.messages.create(
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,  # <-- system prompt
            messages=messages,  # <-- user prompt
            **kwargs,
        )
        if response == "Output blocked by content filtering policy":
            return None
        return response.content[0].text

    def generate(
        self, messages, use_cache=1, max_tokens=500, temperature=1e-5, **kwargs
    ):
        if use_cache == 1:
            response = self.api_with_cache.generate(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )
        else:
            response = self._generate(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )
        return response


class LocalModelWrapper(LLMWrapper):
    def __init__(self, model, path_name=None, use_vllm=False, **kwargs):
        super().__init__(
            model,
        )
        self._setup(model, use_vllm=use_vllm, path_name=path_name, **kwargs)

    def _setup(
        self,
        model,
        path_name=None,
        max_retry=30,
        use_vllm=False,
        **kwargs,
    ):
        if path_name is None:
            path_name = model

        if use_vllm:
            local_model = vllm.LLM(
                model=path_name,
                tensor_parallel_size=torch.cuda.device_count(),
                **kwargs,
            )
        else:
            local_model = pipeline(
                "text-generation",
                device_map="auto",
                model=path_name,
                model_kwargs=kwargs,
            )

        # TODO: Add cache support

        self.api = local_model
        self.api_with_cache = LocalModelAPICache(
            client=local_model, port=PORT, max_retry=max_retry
        )
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate

    def generate(
        self, messages, use_cache=1, max_tokens=500, temperature=1e-5, **kwargs
    ):
        if use_cache == 1:
            return self.api_with_cache.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        return self._generate(messages, max_tokens, temperature, **kwargs)

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens=500,
        temperature=1e-5,
        max_concurrent=3,
        **kwargs,
    ):
        if len(messages) == 0:
            return []
        if isinstance(self.api, vllm.LLM):
            return self._vllm_generate(messages, max_tokens, temperature, **kwargs)
        else:
            output = self.api(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return [o[0]["generated_text"][-1]["content"] for o in output]

    def batched_generate(
        self,
        messages: List[Dict[str, str]],
        use_cache=1,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        if use_cache == 1:
            return self.api_with_cache.batched_generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        return self._batched_generate(messages, max_tokens, temperature, **kwargs)

    def _generate(self, messages, max_tokens=500, temperature=1e-5, **kwargs):
        if isinstance(self.api, vllm.LLM):
            return self._vllm_generate([messages], max_tokens, temperature, **kwargs)[0]
        else:
            output = self.api(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return output[0]["generated_text"][-1]["content"]

    def _vllm_generate(
        self, messages: List[Dict[str, str]], max_tokens=500, temperature=1e-5, **kwargs
    ):
        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        tokenizer = self.api.get_tokenizer()
        formatted_prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        output = self.api.generate(formatted_prompts, sampling_params)
        return [o.outputs[0].text for o in output]
