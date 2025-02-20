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

from . import llm_wrapper_register
from .base import LLMWrapper
from .rate_limiter import RateLimiter
from ..LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from ..tasks import BaseTask

MODEL_COSTS = {
    'gpt-4o-mini': {
        'input': 0.15,
        'output': 0.6
    },
    'gpt-4o': {
        'input': 2.5,
        'output': 10
    },
    'o1': {
        'input': 15,
        'output': 60
    },
    'o3-mini': {
        'input': 1.1,
        'output': 4.4
    }

}
@llm_wrapper_register.register("gpt")
class GPTWrapper(LLMWrapper):
    exceptions_to_catch = (
        openai.RateLimitError,
        openai.APIError,
        openai.APITimeoutError,
    )

    def __init__(
        self,
        model,
        max_retry=30,
        min_backoff=1.0,
        max_backoff=60.0,
        port=6832,
        timeout=20,
        redis_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__(
            model,
            max_retry=max_retry,
            min_backoff=min_backoff,
            max_backoff=max_backoff,
        )
        self.timeout = timeout
        self.api = OpenAI()
        self.api_with_cache = OpenAIAPICache(port=port, **redis_kwargs)
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate
        self.total_cost = 0

    def get_cost(self):
        return self.total_cost
    
    def reset_cost(self):
        self.total_cost = 0
        
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

        async def _async_generate(sem, **kwargs):
            async with sem:
                for _ in range(self.max_retry):
                    try:
                        resp = await client.chat.completions.create(timeout=self.timeout, **kwargs)
                        status_bar.update(1)
                        self.rate_limiter.add_event()
                        return resp
                    except self.exceptions_to_catch as e:
                        self.rate_limiter.backoff(e)
                        continue
                raise Exception(
                    "Max retry exceeded and failed to get response from API, possibly due to bad API requests."
                )

        self.rate_limiter.add_event()
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
        
        for r in resp:
            input_cost = r.usage.prompt_tokens * MODEL_COSTS[model]['input'] / 1000000
            output_cost = r.usage.completion_tokens * MODEL_COSTS[model]['output'] / 1000000
            self.total_cost += input_cost + output_cost
            
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
        self.rate_limiter.add_event()
        for _ in range(self.max_retry):
            try:
                resp = self.api.chat.completions.create(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                    timeout=self.timeout,
                    **kwargs,
                )
                input_cost = resp.usage.prompt_tokens * MODEL_COSTS[model]['input'] / 1000000
                output_cost = resp.usage.completion_tokens * MODEL_COSTS[model]['output'] / 1000000
                self.total_cost += input_cost + output_cost
                return resp.choices[0].message.content
            except self.exceptions_to_catch as e:
                self.rate_limiter.backoff(e)
                continue
        raise Exception(
            "Max retry exceeded and failed to get response from API, possibly due to bad API requests."
        )
