from ..register import Register

llm_wrapper_register = Register(name="llm_wrapper")

from .base import LLMWrapper
from .claude import ClaudeWrapper
from .gpt import GPTWrapper
from .local import LocalModelWrapper, LocalHFWrapper, LocalVllmWrapper
