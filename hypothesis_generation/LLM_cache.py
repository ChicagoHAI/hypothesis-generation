""" A redis-based cache wrapper for GPT-3/Jurassic-1 API to avoid duplicate requests """

""" Modified based on Yiming Zhang's implementation for openai api cache"""
import hashlib
import collections
import pickle
import logging
import time
import threading
from abc import ABC

import redis

import anthropic

logger = logging.getLogger(name="LLM_cache")


def deterministic_hash(data) -> int:
    try:
        data_str = str(data).encode("utf-8")
    except:
        raise Exception(f"Unable to convert type {type(data)} to string.")
    return int(hashlib.sha512(data_str).hexdigest(), 16)


class FrozenDict:
    """frozen, hashable mapping"""

    def __init__(self, mapping):
        self.data = {}
        for key, value in mapping.items():
            if not isinstance(key, collections.abc.Hashable):
                raise Exception(f"{type(key)} is not hashable")
            if not isinstance(value, collections.abc.Hashable):
                if isinstance(value, collections.abc.Mapping):
                    value = FrozenDict(value)
                elif isinstance(value, collections.abc.Sequence):
                    value = tuple(value)
                else:
                    raise Exception(f"{type(value)} is not hashable")
            self.data[key] = value

    def __hash__(self):
        ordered_keys = sorted(self.data.keys(), key=deterministic_hash)
        return deterministic_hash(tuple((k, self.data[k]) for k in ordered_keys))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        raise Exception("FrozenDict is immutable")

    def __repr__(self):
        return repr(self.data)

    def __eq__(self, other):
        return self.data == other.data


class RateLimiter:
    def __init__(self):
        self.min_backoff = self.backoff_time = 1.0
        self.max_backoff = 60.0
        self.lock = threading.Lock()

    def backoff(self):
        logger.warning(f"Backing off for {self.backoff_time:.1f} seconds")
        time.sleep(self.backoff_time)
        with self.lock:
            self.backoff_time = min(self.backoff_time * 2, self.max_backoff)
        logger.debug(f"Setting backoff time to {self.backoff_time:.1f} seconds")

    def add_event(self):
        with self.lock:
            self.backoff_time = max(self.min_backoff, self.backoff_time * 0.75)


class APICache(ABC):
    """Abstract base class for other cache wrappers.

    Should not be instantiated on its own."""

    service = ""
    exceptions_to_catch = tuple()

    def __init__(self, max_retry=30, **redis_kwargs: dict):
        self.r = redis.Redis(host="localhost", **redis_kwargs)

        # max 60 requests per 60 seconds
        self.rate_limiter = RateLimiter()
        self.costs = []
        self.max_retry = max_retry

    def api_call(self, *args, **kwargs):
        raise NotImplementedError("api_call() is not implemented")

    def generate(self, overwrite_cache: bool = False, **kwargs):
        """Makes an API request if not found in cache, and returns the response.

        Args:
            overwrite_cache: If true, ignore and overwrite existing cache.
              Useful when sampling multiple times.
            **kwargs: Generation specific arguments passed to the API.

        Returns:
            A JSON-like API response.
        """
        query = FrozenDict(kwargs)
        hashval = hash(query)
        cache = self.r.hget(hashval, "data")
        if overwrite_cache:
            logger.debug("Overwriting cache")
        elif cache is not None:
            query_cached, resp_cached = pickle.loads(cache)
            if query_cached == query:
                logger.debug(f"Matched cache for query")
                return resp_cached
            logger.debug(
                f"Hash matches for query and cache, but contents are not equal. "
                + "Overwriting cache."
            )
        else:
            logger.debug(f"Matching hash not found for query")

        self.rate_limiter.add_event()
        logger.debug(f"Request Completion from {self.service} API...")

        for _ in range(self.max_retry):
            try:
                resp = self.api_call(**kwargs)
                break
            except self.exceptions_to_catch as e:
                logger.warning(
                    f"Getting an {type(e).__name__} from API, backing off..."
                )
                self.rate_limiter.backoff()
            except anthropic.BadRequestError as e:
                resp = "Output blocked by content filtering policy"
                break

        data = pickle.dumps((query, resp))
        logger.debug(f"Writing query and resp to Redis")
        self.r.hset(hashval, "data", data)

        return resp


class OpenAIAPICache(APICache):
    """A cache wrapper for OpenAI's Chat and Completion API calls.

    Typical usage example:

      api = OpenAIAPICache(open("key.txt").read().strip(), 6379)
      resp = api.generate(model="text-davinci-002", prompt="This is a test", temperature=0.0)
    """

    import openai

    service = "OpenAI"
    exceptions_to_catch = (
        openai.RateLimitError,
        openai.APIError,
        openai.APITimeoutError,
    )

    def __init__(self, mode: str = "completion", **redis_kwargs: dict):
        """Initializes an OpenAIAPICache Object.

        Args:
            port: Port of the Redis backend.
            mode: "completion" or "chat", determines which API to call
        """
        self.mode = mode
        if mode == "completion":
            self.api_call = self.openai.Completion.create
        elif mode == "chat":
            self.api_call = self.openai.ChatCompletion.create
        super().__init__(**redis_kwargs)


class ClaudeAPICache(APICache):
    """A cache wrapper for Anthropic Message API calls."""

    service = "Claude"
    exceptions_to_catch = (
        anthropic.RateLimitError,
        # TODO: add more exceptions
    )

    def __init__(self, client, **redis_kwargs: dict):
        """Initializes an ClaudeAPICache Object.

        Args:
            port: Port of the Redis backend.
            client: Authenticated Claude client
        """
        self.claude = client
        self.api_call = self.claude.messages.create
        super().__init__(**redis_kwargs)


class LocalModelAPICache(APICache):
    """A cache wrapper for Anthropic Message API calls."""

    service = "LocalModel"
    exceptions_to_catch = (
        # TODO: add more exceptions
    )

    def __init__(self, client, **redis_kwargs: dict):
        """Initializes an LocalModelAPICache Object.

        Args:
            port: Port of the Redis backend.
            client: intiailzed LocalModel
        """
        self.localmodel = client
        # TODO: pipeline has no generate method
        self.api_call = self.localmodel.generate
        super().__init__(**redis_kwargs)
