GPT_MODELS = {
    'turbo35_0613': 'gpt-3.5-turbo-0613',
    'turbo35_1106': 'gpt-3.5-turbo-1106',
    'turbo4': 'gpt-4-1106-preview',
    'gpt-4o-mini': 'gpt-4o-mini',
}

CLAUDE_MODELS = {
    "claude_2": "claude-2.1",
    "claude-2.1": "claude-2.1",
    "claude-2.0": "claude-2.0",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-opus": "claude-3-opus-20240229"
}

LLAMA_MODELS = [
    'Llama-2-7b',
    'Llama-2-7b-chat',
    'Llama-2-13b',
    'Llama-2-13b-chat',
    'Llama-2-70b',
    'Llama-2-70b-chat',
]

MISTRAL_MODELS = [
    'Mixtral-8x7B',
    'Mistral-7B'
]

VALID_MODELS = list(GPT_MODELS.keys()) + list(CLAUDE_MODELS.keys()) + LLAMA_MODELS + MISTRAL_MODELS


INST_WRAPPER = {
    'llama': ['',''],
    'default':['###\nInstruction:\n','###\n'],
    'mistral':['[INST] ','[/INST]\n']
}