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

from .rate_limiter import RateLimiter
from ..LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from ..tasks import BaseTask


class LLMWrapper(ABC):
    def __init__(
        self,
        model,
        max_retry=30,
        min_backoff=1.0,
        max_backoff=60.0,
    ):
        self.model = model
        self.max_retry = max_retry
        self.rate_limiter = RateLimiter(
            min_backoff=min_backoff,
            max_backoff=max_backoff,
        )

    @abstractmethod
    def _generate(
        self,
        messages: List[Dict[str, str]],
        model,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def _batched_generate(
        self,
        messages: List[List[Dict[str, str]]],
        model: str,
        max_concurrent=3,
        **kwargs,
    ) -> List[str]:
        pass

    def generate(
        self,
        messages: List[Dict[str, str]],
        cache_seed=None,
        **kwargs,
    ):
        if cache_seed is not None:
            return self.api_with_cache.generate(
                messages=messages,
                model=self.model,
                cache_seed=cache_seed,
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
        cache_seed=None,
        **kwargs,
    ):
        if len(messages) == 1:
            return [self.generate(messages[0], cache_seed=cache_seed, **kwargs)]

        if cache_seed is not None:
            return self.api_with_cache.batched_generate(
                messages=messages,
                model=self.model,
                max_concurrent=max_concurrent,
                cache_seed=cache_seed,
                **kwargs,
            )
        return self._batched_generate(
            messages,
            model=self.model,
            max_concurrent=max_concurrent,
            **kwargs,
        )
