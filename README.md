# Hypothesis Generation with Large Language Models

![](hypogenic_figure1_large_font.jpg)

Welcome to the GitHub repository for our paper, ["Hypothesis Generation with Large Language Models"](https://arxiv.org/abs/2404.04326). This repository is dedicated to the exploration and development of novel methodologies using large language models (LLMs) to generate hypotheses, a foundational element of scientific progress. Our work, presented in detail in the accompanying paper, highlights the capability of LLMs not just to assist but to innovate in the hypothesis generation process for scientific inquiry.


## Install environment
You can directly install HypoGeniC using the following commands:
```
conda create --name hypogenic
pip install hypogenic
```
OR
```
conda create --name hypogenic
pip install -r requirements.txt
```

## Set up path

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

### 1. [Optional] Start Redis server
```bash
PORT=<port_number>
cd $PATH_PREFIX/redis-stable/src
./redis-server --port $PORT
```

### 2. Hypothesis Generation
```bash
hypogenic_generation --args
```

### 3. Hypothesis Inference
```bash
hypogenic_inference --args
```

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
    



