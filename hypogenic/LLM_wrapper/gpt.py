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
        redis_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__(
            model,
            max_retry=max_retry,
            min_backoff=min_backoff,
            max_backoff=max_backoff,
        )
        self.api = OpenAI()
        self.api_with_cache = OpenAIAPICache(port=port, **redis_kwargs)
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

        async def _async_generate(sem, **kwargs):
            async with sem:
                for _ in range(self.max_retry):
                    try:
                        resp = await client.chat.completions.create(**kwargs)
                        status_bar.update(1)
                        self.rate_limiter.add_event()
                        return resp
                    except self.exceptions_to_catch as e:
                        self.rate_limiter.backoff(error_msg=str(e))
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
                    **kwargs,
                )
                return resp.choices[0].message.content
            except self.exceptions_to_catch as e:
                self.rate_limiter.backoff(error_msg=str(e))
                continue
        raise Exception(
            "Max retry exceeded and failed to get response from API, possibly due to bad API requests."
        )
