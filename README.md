# Hypothesis Generation with Large Language Models

![hypogenic_figure1_large_font.jpg](https://raw.githubusercontent.com/ChicagoHAI/hypothesis-generation/master/hypogenic_figure1_large_font.jpg)

Welcome to the GitHub repository for our paper, ["Hypothesis Generation with Large Language Models"](https://arxiv.org/abs/2404.04326). This repository is dedicated to the exploration and development of novel methodologies using large language models (LLMs) to generate hypotheses, a foundational element of scientific progress. Our work, presented in detail in the accompanying paper, highlights the capability of LLMs not just to assist but to innovate in the hypothesis generation process for scientific inquiry.

## Table of Contents
- [Install environment](#install-environment)
- [Optional: set up Redis server for caching LLM responses](#optional-set-up-redis-server-for-caching-llm-responses)
- [Usage](#usage)
  - [Optional: Start Redis server](#optional-start-redis-server)
  - [Hypothesis Generation](#hypothesis-generation)
  - [Hypothesis Inference](#hypothesis-inference)
- [Use HypoGeniC in your code](#use-hypogenic-in-your-code)
- [Add a new task or dataset](#add-a-new-task-or-dataset)
  - [Data preprocessing](#data-preprocessing)
  - [Write config.yaml](#write-configyaml)
  - [Examples](#examples)
  - [Write an extract_label function for your new task](#write-an-extract_label-function-for-your-new-task)

## Install environment
You can directly install HypoGeniC using the following commands:
```bash
conda create --name hypogenic python=3.10
conda activate hypogenic

pip install hypogenic
```
OR

**We recommend using the following installation procedure for easy update and customizability**
```bash
git clone https://github.com/ChicagoHAI/hypothesis-generation.git
cd hypothesis-generation

conda create --name hypogenic python=3.10
conda activate hypogenic

pip install -r requirements.txt
pip install -e .
```

## [Optional]: set up [Redis](https://redis.io) server for caching LLM responses
To save computation or API cost, we use Redis server to cache prompt & response pairs.

Install Redis server from source using the following commands:
Note: Please install in the directory of `PATH_PREFIX`.
```bash
wget https://download.redis.io/redis-stable.tar.gz
tar -xzvf redis-stable.tar.gz
cd redis-stable
make
```

## Usage

The datasets used in our paper is at [HypoGeniC-datasets](https://github.com/ChicagoHAI/HypoGeniC-datasets)

For replicating the results in the paper, you can follow the steps below:
### 1. [Optional] Start Redis server

The default port used by HypoGeniC is 6832. If you want to use a different port, please specify the port number in the `--port` argument.
```bash
cd $PATH_PREFIX/redis-stable/src
./redis-server --port 6832
```

### 2. Hypothesis Generation

For help with the arguments, run:
```bash
hypogenic_generation --help
```
### 3. Hypothesis Inference

For help with the arguments, run:
```bash
hypogenic_inference --help
```
**We will support command lines for HypoGeniC on new tasks and datasets in a later release.**

## Use HypoGeniC in your code
To use HypoGeniC with you own code, tasks, and datasets, you can follow the steps below:
```bash
git clone https://github.com/ChicagoHAI/HypoGeniC-datasets.git ./data
python ./examples/generation.py
```

More examples can be found in `examples/` directory.

## Add a new task or dataset

### 1. Data preprocessing
- To use HypoGeniC, we require users to provide a dataset in the HuggingFace datasets format:
    - `<TASK>_train.json`: A json file containing the training data. 
    - `<TASK>_test.json`: A json file containing the test data. 
    - `<TASK>_val.json`: A json file containing the validation data. 
    - The json file should have keys: `'text_features_1'`, ... `'text_features_n'`, `'label'`. The values corresponding to each key should be a list of strings.

### 2. Write config.yaml
Create the `config.yaml` file in the same directory as the dataset. In the `config.yaml` file, please specify the following fields:
```yaml
task_name: <TASK>

train_data_path: ./<TASK>_train.json
val_data_path: ./<TASK>_test.json
test_data_path: ./<TASK>_val.json

prompt_templates:
  # You can use keys in your dataset as placeholders in the prompt templates
  #   For example, if your dataset has a key 'text_features_1', you can use it as ${text_features_1}
  EXTRA_KEY1: <VALUES>
  EXTRA_KEY2: <VALUES>
  # ...

  # You can use EXTRA_KEYs above as placeholders in the prompt templates
  #   For example, You can use ${EXTRA_KEY1} in the prompt templates
  # Additionally, you can use the following placeholders in the prompt templates
  #   ${num_hypotheses}: Number of hypotheses to generate
  # The prompt templates are formatted as follows:
  #   [
  #     {"role": "role1", "content": "<ROLE1_PROMPT_TEMPLATE>"}, 
  #     {"role": "role2", "content": "<ROLE2_PROMPT_TEMPLATE>"},
  #     ...
  #   ]
  batched_generation:
    role1: <ROLE1_PROMPT_TEMPLATE>
    role2: <ROLE2_PROMPT_TEMPLATE>
    # ...
  few_shot_baseline:
    role1: <ROLE1_PROMPT_TEMPLATE>
    role2: <ROLE2_PROMPT_TEMPLATE>
    # ...
  inference:
    role1: <ROLE1_PROMPT_TEMPLATE>
    role2: <ROLE2_PROMPT_TEMPLATE>
    # ...
  is_relevant:
    role1: <ROLE1_PROMPT_TEMPLATE>
    role2: <ROLE2_PROMPT_TEMPLATE>
    # ...
  adaptive_inference:
    role1: <ROLE1_PROMPT_TEMPLATE>
    role2: <ROLE2_PROMPT_TEMPLATE>
    # ...
  adaptive_selection:
    role1: <ROLE1_PROMPT_TEMPLATE>
    role2: <ROLE2_PROMPT_TEMPLATE>
    # ...
```

### Examples

`./headline_binary/headline_binary_test.json`

```json
{
  "headline_1": [
    "What Up, Comet? You Just Got *PROBED*",
    "..."
  ],
  "headline_2": [
    "Scientists Everywhere Were Holding Their Breath Today. Here's Why.",
    "..."
  ],
  "label": [
    "Headline 2 has more clicks than Headline 1",
    "..."
  ]
}
```

`./headline_binary/config.yaml`

```yaml
task_name: headline_binary

train_data_path: ./headline_binary_train.json
val_data_path: ./headline_binary_test.json
test_data_path: ./headline_binary_val.json

prompt_templates:
  observations: |
    Headline 1: ${headline_1}
    Headline 2: ${headline_2}
    Observation: ${label}

  # More EXTRA_KEYs

  batched_generation:
    system: |-
      ...
      Please propose ${num_hypotheses} possible hypotheses and generate them in the format of 1. [hypothesis], 2. [hypothesis], ... ${num_hypotheses}. [hypothesis].

    user: |-
      Here are the Observations:
      ${observations}

      Please generate hypotheses that can help determine which headlines have more clicks.
      Please propose ${num_hypotheses} possible hypotheses.

      Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... ${num_hypotheses}. [hypothesis]. 

      Proposed hypotheses:

  # few_shot_baseline
  # inference
  # is_relevant
  # adaptive_inference
  # adaptive_selection
```
### 3. Write an extract_label function for your new task
As we show in `examples/generation.py`, you can create a new task by using our `BaseTask` constructor (line 63). You need to implement the `extract_label` function for your new task. The `extract_label` function should take a string input (LLM generated inference text), and return the label extracted from the input. 

**If no `extract_label` function is provided, the default version will be used, which looks for `final answer:\s+<begin>(.*)<end>` in the LLM generated text.**

**Note: you need to make sure the extracted label are in same format with the `'label'` in your dataset, since the extracted label will be compared with the true label to check correctness of each LLM inference.**