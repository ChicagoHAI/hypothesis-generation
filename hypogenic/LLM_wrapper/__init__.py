from ..register import Register
from ..logger_config import LoggerConfig

llm_wrapper_register = Register(name="llm_wrapper")

from .base import LLMWrapper
from .claude import ClaudeWrapper
from .gpt import GPTWrapper

try:
    from .local import LocalModelWrapper, LocalHFWrapper, LocalVllmWrapper
except ImportError as e:
    # vllm or other local model dependencies not available
    logger = LoggerConfig.get_logger("LLM_wrapper")
    logger.warning(f"Could not import local model wrappers.\n You can disregard this warning if you are only using api-based models.\n")
