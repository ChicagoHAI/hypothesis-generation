# Model costs per 1M tokens (in USD)
MODEL_COSTS = {
    # OpenAI models
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
    },
    # OpenRouter model costs
    'openai/gpt-4o-mini': {
        'input': 0.15,
        'output': 0.6
    },
    'openai/gpt-4o': {
        'input': 2.5,
        'output': 10
    },
    'anthropic/claude-3.5-sonnet': {
        'input': 3.0,
        'output': 15.0
    },
    'anthropic/claude-3-haiku': {
        'input': 0.25,
        'output': 1.25
    },
    'meta-llama/llama-3.1-70b-instruct': {
        'input': 0.1,
        'output': 0.28
    },
    # Qwen models
    'qwen/qwen3-32b': {
        'input': 0.03,
        'output': 0.13
    }
}

from ..logger_config import LoggerConfig
logger = LoggerConfig.get_logger("Model_Costs")
def get_model_cost(model_name):
    """Get cost information for a model, return None if not found."""
    return MODEL_COSTS.get(model_name)

def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculate cost for a model given input and output tokens."""
    costs = get_model_cost(model_name)
    if not costs:
        logger.warning(f"Cost information not found for model: {model_name}, using 0 cost.")
        return 0  # Return 0 if cost not available

    input_cost = input_tokens * costs['input'] / 1000000
    output_cost = output_tokens * costs['output'] / 1000000
    return input_cost + output_cost