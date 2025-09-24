from abc import ABC, abstractmethod
import pickle
import math
import json
from typing import Callable, Dict, List
import torch
import re
import os
import numpy as np
import random
import openai

import asyncio
import tqdm
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic, Anthropic
import httpx

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
from .model_costs import calculate_cost
from ..LLM_cache import ClaudeAPICache, LocalModelAPICache, OpenAIAPICache
from ..tasks import BaseTask
from ..logger_config import LoggerConfig

@llm_wrapper_register.register("gpt")
class GPTWrapper(LLMWrapper):
    exceptions_to_catch = (
        openai.RateLimitError,
        openai.APIError,
        openai.APITimeoutError,
        json.JSONDecodeError,  # Handle malformed API responses from OpenRouter/OpenAI
        ConnectionError,       # Network issues
        TimeoutError,         # Generic timeout errors
        httpx.HTTPError,      # HTTP-related errors from underlying httpx client
        httpx.ConnectError,   # Connection failures
        httpx.ReadTimeout,    # Read timeout errors
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
        use_openrouter=False,
        **kwargs,
    ):
        super().__init__(
            model,
            max_retry=max_retry,
            min_backoff=min_backoff,
            max_backoff=max_backoff,
        )
        self.timeout = timeout
        self.use_openrouter = use_openrouter

        # Initialize OpenAI client with OpenRouter support
        client_kwargs = {}
        if use_openrouter:
            openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
            if not openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable is required when use_openrouter=True")
            client_kwargs['base_url'] = 'https://openrouter.ai/api/v1'
            client_kwargs['api_key'] = openrouter_api_key

        self.api = OpenAI(**client_kwargs)
        self.api_with_cache = OpenAIAPICache(port=port, **redis_kwargs)
        self.api_with_cache.api_call = self._generate
        self.api_with_cache.batched_api_call = self._batched_generate
        self.total_cost = 0

        # Store client kwargs for async client initialization
        self._client_kwargs = client_kwargs

    def get_cost(self):
        return self.total_cost
    
    def reset_cost(self):
        self.total_cost = 0

    def _preprocess_messages_for_model(self, messages, model):
        """Preprocess messages based on model-specific requirements."""
        # Check if this is a Qwen model that needs /no_think
        if 'qwen3-32b' in model.lower():
            # Create a copy to avoid modifying the original
            processed_messages = []
            for msg in messages:
                if msg.get('role') == 'user':
                    # Prepend /no_think\n to user messages
                    new_msg = msg.copy()
                    new_msg['content'] = f"/no_think\n{msg['content']}"
                    processed_messages.append(new_msg)
                else:
                    processed_messages.append(msg)
            return processed_messages
        return messages

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

        # Preprocess messages for model-specific requirements
        messages = [self._preprocess_messages_for_model(msg_list, model) for msg_list in messages]

        client = AsyncOpenAI(**self._client_kwargs)
        status_bar = tqdm.tqdm(total=len(messages))

        async def _async_generate(sem, **kwargs):
            async with sem:
                logger = LoggerConfig.get_logger("GPTWrapper")
                for retry_count in range(self.max_retry):
                    try:
                        resp = await client.chat.completions.create(timeout=self.timeout, **kwargs)
                        status_bar.update(1)
                        self.rate_limiter.add_event()
                        return resp
                    except self.exceptions_to_catch as e:
                        logger.warning(
                            f"API request failed (attempt {retry_count + 1}/{self.max_retry}): {type(e).__name__}: {str(e)}"
                        )
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
            cost = calculate_cost(model, r.usage.prompt_tokens, r.usage.completion_tokens)
            self.total_cost += cost

        # Extract responses and log them in debug mode
        responses = [r.choices[0].message.content for r in resp]
        logger = LoggerConfig.get_logger("GPTWrapper")
        for i, response in enumerate(responses):
            logger.debug(f"Batch response {i+1}: {response}")

        return responses

    def _generate(
        self,
        messages,
        model: str,
        max_tokens=500,
        temperature=1e-5,
        n=1,
        **kwargs,
    ):
        # Preprocess messages for model-specific requirements
        messages = self._preprocess_messages_for_model(messages, model)

        self.rate_limiter.add_event()
        logger = LoggerConfig.get_logger("GPTWrapper")
        for retry_count in range(self.max_retry):
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
                cost = calculate_cost(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
                self.total_cost += cost
                response_content = resp.choices[0].message.content
                logger.debug(f"Model response: {response_content}")
                return response_content
            except self.exceptions_to_catch as e:
                logger.warning(
                    f"API request failed (attempt {retry_count + 1}/{self.max_retry}): {type(e).__name__}: {str(e)}"
                )
                self.rate_limiter.backoff(e)
                continue
        raise Exception(
            "Max retry exceeded and failed to get response from API, possibly due to bad API requests."
        )


@llm_wrapper_register.register("openrouter")
class OpenRouterWrapper(GPTWrapper):
    """OpenRouter wrapper that automatically configures GPTWrapper for OpenRouter usage."""

    def __init__(self, model, **kwargs):
        # Force use_openrouter=True for this wrapper
        kwargs['use_openrouter'] = True
        super().__init__(model=model, **kwargs)
