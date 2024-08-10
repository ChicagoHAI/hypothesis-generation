from abc import ABC, abstractmethod
import pickle
import math
from typing import Callable, Dict, List
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

from .register import Register
from .LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from .tasks import BaseTask

llm_wrapper_register = Register(name="llm_wrapper")


class LLMWrapper(ABC):
    def __init__(self, model):
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        use_cache=1,
        **kwargs,
    ):
        if use_cache == 1:
            return self.api_with_cache.generate(
                messages=messages,
                model=self.model,
                **kwargs,
            )
        return self._generate(
            messages,
            model=self.model,
            **kwargs,
        )

    def batched_generate(
        self,
        messages: List[List[Dict[str, str]]],
        max_concurrent=3,
        use_cache=1,
        **kwargs,
    ):
        if use_cache == 1:
            return self.api_with_cache.batched_generate(
                messages=messages,
                model=self.model,
                max_concurrent=max_concurrent,
                **kwargs,
            )
        return self._batched_generate(
            messages,
            model=self.model,
            max_concurrent=max_concurrent,
            **kwargs,
        )


@llm_wrapper_register.register("gpt")
class GPTWrapper(LLMWrapper):
    def __init__(self, model, max_retry=30, port=6832, **kwargs):
        super().__init__(model)
        self.api = OpenAI()
        self.api_with_cache = OpenAIAPICache(port=port, max_retry=max_retry, **kwargs)
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate

    def _batched_generate(
        self,
        messages: List[List[Dict[str, str]]],
        model: str,
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
        async def _async_generate(sem, **kwargs):
            async with sem:
                resp = await client.chat.completions.create(**kwargs)
                status_bar.update(1)
                return resp

        sem = asyncio.Semaphore(max_concurrent)
        tasks = [
            _async_generate(
                sem,
                messages=messages[i],
                model=model,
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

    def _generate(
        self,
        messages,
        model: str,
        max_tokens=500,
        temperature=1e-5,
        n=1,
        **kwargs,
    ):
        resp = self.api.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            **kwargs,
        )
        return resp.choices[0].message.content


@llm_wrapper_register.register("claude")
class ClaudeWrapper(LLMWrapper):
    def __init__(self, model, max_retry=30, port=6832, **kwargs):
        super().__init__(model)
        self.api = Anthropic()
        self.api_with_cache = ClaudeAPICache(
            client=self.api, port=port, max_retry=max_retry, **kwargs
        )
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens=500,
        temperature=1e-5,
        max_concurrent=3,
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
                messages=messages[i],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            for i in range(len(messages))
        ]
        loop = asyncio.get_event_loop()
        resp = loop.run_until_complete(asyncio.gather(*tasks))
        return [r.content[0].text for r in resp]

    def _generate(
        self,
        messages,
        model: str,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        for idx, msg in enumerate(messages):
            if msg["role"] == "system":
                system_prompt = messages.pop(idx)["content"]
                break

        response = self.api.messages.create(
            system=system_prompt,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        if response == "Output blocked by content filtering policy":
            return None
        return response.content[0].text


class LocalModelWrapper(LLMWrapper):
    def __init__(
        self,
        model,
        model_constructor: Callable,
        path_name=None,
        max_retry=30,
        port=6832,
        **kwargs,
    ):
        super().__init__(model)
        if path_name is None:
            path_name = model

        local_model = model_constructor(model=path_name, **kwargs)

        self.api = local_model
        self.api_with_cache = LocalModelAPICache(
            client=local_model, port=port, max_retry=max_retry
        )
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        raise NotImplementedError

    def _generate(
        self,
        messages,
        model: str,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        raise NotImplementedError


@llm_wrapper_register.register("huggingface")
class LocalHFWrapper(LocalModelWrapper):
    def __init__(self, model, path_name=None, max_retry=30, port=6832, **kwargs):
        super().__init__(
            model=model,
            model_constructor=pipeline,
            path_name=path_name,
            max_retry=max_retry,
            port=port,
            # pipeline kwargs
            task="text-generation",
            device_map="auto",
            model_kwargs=kwargs,
        )

    def _batched_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        if len(messages) == 0:
            return []
        output = self.api(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return [o[0]["generated_text"][-1]["content"] for o in output]

    def _generate(
        self,
        messages,
        model: str,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        output = self.api(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return output[0]["generated_text"][-1]["content"]


@llm_wrapper_register.register("vllm")
class LocalVllmWrapper(LocalModelWrapper):
    def __init__(
        self,
        model,
        path_name=None,
        max_retry=30,
        port=6832,
        **kwargs,
    ):
        super(__class__, self).__init__(
            model=model,
            model_constructor=vllm.LLM,
            path_name=path_name,
            max_retry=max_retry,
            port=port,
            # VLLM kwargs
            tensor_parallel_size=torch.cuda.device_count(),
            **kwargs,
        )

    def _generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
    ):
        return self._batched_generate(
            [messages],
            max_tokens,
            temperature,
            **kwargs,
        )[0]

    def _batched_generate(
        self,
        messages: List[List[Dict[str, str]]],
        model: str,
        max_concurrent=3,
        max_tokens=500,
        temperature=1e-5,
        **kwargs,
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
