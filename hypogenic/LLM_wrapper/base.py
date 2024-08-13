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
