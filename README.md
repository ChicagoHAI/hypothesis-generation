<a name="readme-top"></a>

<div align="center">
<img src="https://github.com/ChicagoHAI/chicagohai.github.io/blob/main/static/avatar.jpg" alt="CHAI lab Logo" width="100">

[![Discord](https://img.shields.io/badge/discord-chat-green?logo=discord)](https://discord.gg/BgkfTvBdbV)

</div>

# Hypothesis Generation with Large Language Models
**[Mar'25]** [HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation](https://arxiv.org/abs/2504.11524) 

**[Oct'24]** [Literature Meets Data: A Synergistic Approach to Hypothesis Generation](https://arxiv.org/abs/2410.17309)     

**[Apr'24]** [Hypothesis Generation with Large Language Models](https://arxiv.org/abs/2404.04326) 

<!-- ![hypogenic_figure1_large_font.jpg](https://raw.githubusercontent.com/ChicagoHAI/hypothesis-generation/master/hypogenic_figure1_large_font.jpg) -->

![hypothesis-agent_figure1_large_font.jpg](https://raw.githubusercontent.com/ChicagoHAI/hypothesis-generation/haokun_dev/hypothesis-agent_figure1_large_font.png)

<!-- **Do we keep the figure1 for hypogenic here or what?** -->

This repository is dedicated to the exploration and development of novel methodologies using large language models (LLMs) to generate hypotheses, a foundational element of scientific progress. Our works introduce frameworks for generating hypotheses with LLMs, specifically **HypoGeniC** (**Hypo**thesis **Gen**eration **i**n **C**ontext) is a data-driven framework that generates hypotheses solely based on given datasets, while **HypoRefine** is a synergistic approach 
that incorporates both existing literature and given datasets in an agentic framework to generate hypotheses. Additionally, modules of two Union methods **Literature∪HypoGeniC** and **Literature∪HypoRefine** are provided that mechanistically combine hypotheses from literature only with hypotheses from our frameworks. 
Our work highlights the capability of LLMs to assist and innovate in the hypothesis generation process for scientific inquiry.

<!-- Welcome to the GitHub repository for our paper, ["Hypothesis Generation with Large Language Models"](https://arxiv.org/abs/2404.04326). This repository is dedicated to the exploration and development of novel methodologies using large language models (LLMs) to generate hypotheses, a foundational element of scientific progress. Our work, presented in detail in the accompanying paper, highlights the capability of LLMs not just to assist but to innovate in the hypothesis generation process for scientific inquiry. -->

## Table of Contents
- [Install environment](#install-environment)
- [Optional: set up Redis server for caching LLM responses](#optional-set-up-redis-server-for-caching-llm-responses)
- [Quickstart](#quickstart)
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

## Quickstart

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/ChicagoHAI/hypothesis-generation.git
cd hypothesis-generation

conda create --name hypogenic python=3.10
conda activate hypogenic

pip install -r requirements.txt
pip install -e .
```

2. Copy the dataset folders to `~/hypothesis-generation/data/`:
```bash
cp -r /path/to/dataset/folders ~/hypothesis-generation/data/
```

3. Run the pipeline:
```bash
conda activate hypogenic
cd hypothesis-generation/
./run_pipeline.sh
```

## Detailed Usage

The datasets used in our works are at [HypoGeniC-datasets](https://github.com/ChicagoHAI/HypoGeniC-datasets).

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
To use **HypoGeniC** with you own code, tasks, and datasets, you can follow the steps below:
```bash
git clone https://github.com/ChicagoHAI/HypoGeniC-datasets.git ./data
python ./examples/generation.py
```

To use **HypoRefine** or Union methods, follow the steps below:  
(There will be 3 hypothesis banks generated: **HypoRefine**, Hypotheses solely from literature, and **Literature∪HypoRefine**.)
```bash
git clone https://github.com/ChicagoHAI/Hypothesis-agent-datasets.git ./data
python ./examples/union_generation.py
```

To run default (best hypothesis) inference on generated hypotheses:
```bash
python ./examples/inference.py
```

To run multiple-hypothesis inference on generated hypotheses:
```bash
python ./examples/multi_hyp_inference.py
```

More examples can be found in `examples/` directory.

## Add a new task or dataset

### 1. Data preprocessing
- To use HypoGeniC, we require users to provide a dataset in the HuggingFace datasets format:
    - `<TASK>_train.json`: A json file containing the training data. 
    - `<TASK>_test.json`: A json file containing the test data. 
    - `<TASK>_val.json`: A json file containing the validation data. 
    - The json file should have keys: `'text_features_1'`, ... `'text_features_n'`, `'label'`. The values corresponding to each key should be a list of strings.

### 2. (optional) Literature PDF preprocessing
For **HypoRefine** or Union methods, it is required for users to provide relevant literature PDFs and preprocess them following the steps below:
1. Add PDF files to the directory: literature/YOUR_TASK_NAME/raw/
2. Run the following lines:
```bash
bash ./modules/run_grobid.sh
```
   If you haven't set up grobid before:
```bash
bash ./modules/setup_grobid.sh
```
   Then:
```bash
cd examples
python pdf_preprocess.py --task_name YOUR_TASK_NAME
```
(We will support automated literature search in a later release.)

### 2. Write config.yaml
Create the `config.yaml` file in the same directory as the dataset. In the `config.yaml` file, please specify the following fields:

**Note: For running a basic generation, you will need to write prompt templates for `observations`, `batched_generation`, and `inference`.**
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

**TODO: Instructions for customizing prompt to be updated**

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
  inference:
    system: |-
      ...
    
    user: |-
      ...
      
  # is_relevant
  # adaptive_inference
  # adaptive_selection
```
### 3. Write an extract_label function for your new task
As we show in `examples/generation.py`, you can create a new task by using our `BaseTask` constructor (line 63). You need to implement the `extract_label` function for your new task. The `extract_label` function should take a string input (LLM generated inference text), and return the label extracted from the input. 

**If no `extract_label` function is provided, the default version will be used, which looks for `final answer:\s+(.*)` in the LLM generated text.**

**Note: you need to make sure the extracted label are in same format with the `'label'` in your dataset, since the extracted label will be compared with the true label to check correctness of each LLM inference.**

## BibTeX
If you used this package, please consider citing the following works.
```
@misc{liu2025hypobenchsystematicprincipledbenchmarking,
      title={HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation}, 
      author={Haokun Liu and Sicong Huang and Jingyu Hu and Yangqiaoyu Zhou and Chenhao Tan},
      year={2025},
      eprint={2504.11524},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.11524}, 
}

@misc{liu2024literaturemeetsdatasynergistic,
      title={Literature Meets Data: A Synergistic Approach to Hypothesis Generation}, 
      author={Haokun Liu and Yangqiaoyu Zhou and Mingxuan Li and Chenfei Yuan and Chenhao Tan},
      year={2024},
      eprint={2410.17309},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.17309}, 
}
    
@inproceedings{zhou2024hypothesisgenerationlargelanguage,
      title={Hypothesis Generation with Large Language Models}, 
      author={Yangqiaoyu Zhou and Haokun Liu and Tejes Srivastava and Hongyuan Mei and Chenhao Tan},
      booktitle = {Proceedings of EMNLP Workshop of NLP for Science},
      year={2024},
      url={https://aclanthology.org/2024.nlp4science-1.10/},
}
```
