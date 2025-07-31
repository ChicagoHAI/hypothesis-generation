import os
import argparse
import zipfile
import csv
import json
from ..logger_config import LoggerConfig

from ..LLM_wrapper import llm_wrapper_register

logger = LoggerConfig.get_logger("Config Generator")


def read_dataset(dataset_path):
    files = [f for f in os.listdir(dataset_path) if (f.endswith('.json') or f.endswith('.csv')) and not f.startswith('metadata') and not f.startswith('gt')]
    
    if not files:
        logger.error("No data file (CSV/JSON) found in dataset.")
        return

    file_path = os.path.join(dataset_path, files[0])
    ext = os.path.splitext(file_path)[1].lower()

    with open(file_path) as f:
        if ext == ".csv":
            data_dict = list(csv.DictReader(f))
        elif ext == ".json":
            data_dict = json.load(f)
        else:
            logger.error("No CSV or JSON data file detected.")
            return    
        
    if isinstance(data_dict, list) and data_dict:
        headers = list(data_dict[0].keys())
    elif isinstance(data_dict, dict):
        headers = list(data_dict.keys())
        
    if "label" in headers:
        label_name = "label"
        headers.remove("label")
        input = headers
    else:
        input = headers[0:-1]
        label_name = headers[-1]

    if isinstance(data_dict, list):
        labels = list(set(row[label_name] for row in data_dict if label_name in row))
    elif isinstance(data_dict, dict):
        labels = list(set(data_dict[label_name])) if label_name in data_dict else []
    
    return files, input, label_name, labels

def generate_template(task, label_name, labels):
    config_template = f'''task_name: {task}
label_name: {label_name}

# Using the names of the files, input their names into the following
# If there are fewer than 3 csv files, skip the missing lines
train_data_path: ./{{file name}}
val_data_path: 
test_data_path: 
ood_data_path: ./{task}_ood.json

prompt_templates:
  observations:
    multi_content: |
        # Display the inputs of the data as stated above in the prompt

        The [task decision] is: ${{{label_name}}}
    
  batched_generation:
    system: |-
      # Explain the instructions and research question and what exactly should be generated
    
      Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... ${{num_hypotheses}}. [hypothesis].
      The hypotheses should analyze [objective of task].

    user: |-
      We have seen some hotel reviews:
      ${{observations}}
      Please generate hypotheses that are useful for predicting [objective of task].
      Propose ${{num_hypotheses}} possible hypotheses. Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... $[num_hypotheses]. [hypothesis].
      Proposed hypotheses:
            
  inference:
    system: |-
      # Use the research question and instructions to generate a prompt that would then be used to generate hypotheses
      From past experiences, you learned a pattern. 
      You need to determine whether each of the patterns holds for [name what the inputs determine], and also predict whether [which of the given labels fit]. 
      Give an answer. The answer should be one of these options ({' or '.join(labels).title()}).
      Give your final answer in the format of "Final answer: answer". Do NOT use markdown format.

    user: |-
      Our learned patterns: ${{hypothesis}}
      # Display the inputs of the data as if it is a new set of data points
      Given the pattern you learned above, give an answer of whether [the stimulus fits with which label].
      Think step by step.
      First step: Consider if the pattern can be applied to the [stimulus].
      Second step: Based on the pattern, is this [stimulus] {' or '.join(labels)}?
      Final step: give your final answer in the format of "Final answer: answer". Do NOT use markdown format.

  multiple_hypotheses_inference:
    system: |-
      # Use the research question and instructions to generate a prompt that would then be used to generate hypotheses
      From past experiences, you learned a pattern. 
      You need to determine whether each of the patterns holds for [name what the inputs determine], and also predict whether [which of the given labels fit]. 
      Give an answer. The answer should be one of these options ({' or '.join(labels).title()}).
      Give your final answer in the format of "Final answer: answer". Do NOT use markdown format.

    user: |-
      Our learned patterns: ${{hypotheses}}
      # Display the inputs of the data as if it is a new set of data points
      Given the pattern you learned above, give an answer of whether [the stimulus fits with which label].
      Think step by step.
      First step: Consider if the pattern can be applied to the [stimulus].
      Second step: Based on the pattern, is this [stimulus] {' or '.join(labels)}?
      Final step: give your final answer in the format of "Final answer: answer". Do NOT use markdown format.
    '''

    return config_template


def generate(mod, mod_name, dataset_path, rq="", instr=""):
    api = llm_wrapper_register.build(mod)(mod_name)
    logger.info("API call created.")

    dataset = os.path.splitext(os.path.basename(dataset_path))[0]
    files, input, label_name, labels = read_dataset(dataset_path)

    example_file = os.path.join("hypogenic", "config_generation", "config_examples", "config.yaml")

    with open(example_file) as f:
        example_text = f.read()

        if example_text is None:
            logger.error("No example file provided.")
            return ""

    template = generate_template(dataset, label_name, labels)

    prompt = f"You are an AI that generates configuration files based on given information and a template.\n\n"

    if rq != "":
        prompt += f"The following is the research question:\n{rq}\n\n"

    if instr != "":
        prompt += f"The following are the instructions:\n{instr}\n\n"

    files_str = "\n".join(files)
    input_str = "\n".join(input)
    labels_str = "\n".join(labels)
    
    prompt += f"""This is the name of the dataset:
{dataset}

These are the names of the file: 
{files_str}

This is the input(s) for the dataset:
{input_str}

This is the name of the label: {label_name}

These are the possible labels:
{labels_str}

Below is a template with some basic information that the output must adhere to:
{template}


Below is an example configuration file for reference:
{example_text}


Please generate a new configuration file based on the research question and instructions.
Only return the content of the configuration file with NO headers or footers.
    """

    logger.info(f"Generated Prompt:\n{prompt}")

    try:
        messages = [{"role": "user", "content": prompt}]

        logger.info(f"Cost before generation: {api.get_cost()} USD")

        response = api.api_with_cache.api_call(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.05,
            max_tokens=5000
        )

        logger.info(f"Cost after generation: {api.get_cost()} USD")

        config = response.strip()

    except Exception as e:
        logger.error(f"Error generating config: {e}")
        return ""
    
    logger.info(f"Generated config file:\n{config}")

    try:
        with open(os.path.join(dataset_path, "config.yaml"), 'w') as f:
            f.write(config)
    
    except Exception as e:
        logger.error(f"Error writing config to file: {e}")
        return ""

    logger.info("Config generation successful")

    return ""

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--research_question", type=str, required=False, default="")
    parser.add_argument("--instructions", type=str, required=False, default="")

    parser.add_argument("--model_type", type=str, required=False, default="gpt")
    parser.add_argument("--model_name", type=str, required=False, default="gpt-4o-mini")

    args = parser.parse_args()

    config = generate(
        args.model_type,
        args.model_name,
        args.dataset_path,
        args.research_question,
        args.instructions
    )