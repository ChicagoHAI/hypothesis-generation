GPT_MODELS = {
    'turbo35_0613': 'gpt-3.5-turbo-0613',
    'turbo35_1106': 'gpt-3.5-turbo-1106',
    'turbo4': 'gpt-4-1106-preview',
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


VALID_MODELS = ['turbo35_0613', 
                'turbo35_1106',
                'turbo4',
                'claude_2',
                ] + LLAMA_MODELS + MISTRAL_MODELS


INST_WRAPPER = {
    'llama': ['',''],
    'default':['###\nInstruction:\n','###\n'],
    'mistral':['[INST] ','[/INST]\n']
}