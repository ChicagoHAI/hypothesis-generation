import os
import yaml
from .LLM_wrapper.gpt import GPTWrapper

BASE_DIR = os.path.join(os.curdir, "hypogenic")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt = GPTWrapper(model="gpt-4o-mini")

def generate(rq, instr):
    example_dir = os.path.join(BASE_DIR, "config_examples")
    example_files = [f for f in os.listdir(example_dir) if os.path.isfile(os.path.join(example_dir, f))]

    examples_text = ""
    for fname in example_files:
        with open(os.path.join(example_dir, fname), 'r') as f:
            content = f.read()
            examples_text += f"### Example: {fname}\n```\n{content}\n```\n\n"

    prompt = f"""You are an AI that generates configuration files based on instructions and a research question.

Research Question:
{rq}

Instructions:
{instr}

Below are several example configuration files for reference:
{examples_text}

Please generate a new configuration file based on the research question and instructions.
Only return the content of the configuration file. Do not include any extra explanation.
    """

    try:
        messages = [{"role": "user", "content": prompt}]

        print("Cost before generation:", gpt.get_cost(), "USD")

        response = gpt.api_with_cache.api_call(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=500
        )

        print("Cost after generation:", gpt.get_cost(), "USD")

        return response.strip()

    except Exception as e:
        print("Error generating config:", e)
        return ""
    
if __name__ == "__main__":
    question = "What aspects of a speech on the floor of the American Congress indicate the party of the speaker?"
    instr = '''You are an expert political scientist. Given some key findings in research papers, we want hypotheses that are useful in predicting the political party of a given speaker based on the speech.

Using the given relevant literatures, please propose 10 possible hypothesis pairs.
These hypotheses should identify specific patterns that occur across the provided speeches.

Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... 10. [hypothesis].
The hypotheses should analyze what kind of speeches are likely to be democrat, republican, or independent.
'''

    config = generate(question, instr)
    file = os.path.expanduser('~/Downloads/config_test_1.yaml')

    with open(file, 'w') as f:
        yaml.dump(config, f, default_style=None)