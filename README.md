# Hypothesis Generation with Large Language Models

![hypogenic_figure1_large_font.jpg](https://raw.githubusercontent.com/ChicagoHAI/hypothesis-generation/master/hypogenic_figure1_large_font.jpg)

Welcome to the GitHub repository for our paper, ["Hypothesis Generation with Large Language Models"](https://arxiv.org/abs/2404.04326). This repository is dedicated to the exploration and development of novel methodologies using large language models (LLMs) to generate hypotheses, a foundational element of scientific progress. Our work, presented in detail in the accompanying paper, highlights the capability of LLMs not just to assist but to innovate in the hypothesis generation process for scientific inquiry.


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
hypogenic_inference --<relevant args>
```
The section below will show you how to make a "hypogenic_inference" file.

**We will support command lines for HypoGeniC on new tasks and datasets in a later release.**

## Use HypoGeniC in your code
To use HypoGeniC with you own code, tasks, and datasets, you can follow the steps below:
```bash
git clone https://github.com/ChicagoHAI/HypoGeniC-datasets.git ./data
python ./examples/generation.py
```

More examples can be found in `examples/` directory.

## Add a new task or dataset

To use hypogenic on your own dataset, you should implement the following 3 components: **(UPDATE: WHAT'S THE PROTOCOL WITH THE PIP VS LOCAL INSTALLATION)**
1. The dataset that you want to use
2. A config.yaml file
3. The file to actually run the algorithm from

We will now go over the nuances of each step.

### 1. Data preprocessing
To use HypoGeniC, we require users to provide a dataset in the HuggingFace datasets format:\
* `<TASK>_train.json`: A json file containing the training data. \
* `<TASK>_test.json`: A json file containing the test data. \
* `<TASK>_val.json`: A json file containing the validation data. \
* The json file should have keys: `'text_features_1'`, ... `'text_features_n'`, `'label'`. The values corresponding to each key should be a list of strings.

For example:
`./headline_binary/headline_binary_test.json`
could look like

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

This format helps us standardize data loading.

It's also important to realize that you need some kind of a label, as hypogenic is a supervised training algorithm.

### 2. Write config.yaml
Create the `config.yaml` file in the same directory as the dataset. 

This file provides configuration for the entire run, namely:
* task name
* dataset paths
* prompt templates
In hypogenic/task.py, you'll read the contents of the file in.  Please refer to the examples for how to format them.


The `config.yaml` file should have a skeleton similar like so:
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

And an example is 
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
```

### 3. Write the algorithm's file

In this script is where you'll actually be running hypogenic from.

You're going to need to create:\
**A local llm wrapper** - hypogenic supports vllm, huggingface (see LLM_wrapper/local), the gpt api (LLM_wrapper/gpt), and the claude api (LLM_wrapper/claude).\
**A Task class** - reads from your yaml file to get the relevant details needed for your run.  See Section 2 to get more details about the yaml file.\
**A Prompt class** - This class helps generate promtps at scale by fitting variables to our template.\
**An Inference class** - Inference measures hypotheses' predictive power on the given task.\
**A Generation class** - We create hypotheses with this class.\
**An Update class** - Update contains the main algorithm loop and will prune hypotheses as the steps progress/\

From there, your can either initialize the hypothesis bank or load an existing one.

For example:
```python
hypotheses_bank = {}
if old_hypothesis_file is None:
    # Initalize the hypothesis bank using the update class and save them
    hypotheses_bank = update_class.batched_initialize_hypotheses(
        num_init,
        init_batch_size=3,
        init_hypotheses_per_batch=10,
        use_cache=0,
    )
    update_class.save_to_json(
        hypotheses_bank,
        sample=num_init,
        seed=seed,
        epoch=0,
    )
else:
    # we can load another hypothesis bank
    dict = load_dict(old_hypothesis_file)
    for hypothesis in dict:
        hypotheses_bank[hypothesis] = dict_to_summary_information(
            dict[hypothesis]
        )
```

Finally, you can control how many epochs you want and use the update class to run the hypogenic algorithm
It's also recommended that you save to json after each iteration.

Here's how it might look:

```python
for epoch in range(1):
  hypotheses_bank = update_class.update(
      current_epoch=epoch,
      hypotheses_bank=hypotheses_bank,
      current_seed=seed,
      use_cache=0,
  )
  update_class.save_to_json(
      hypotheses_bank,
      sample="final",
      seed=seed,
      epoch=epoch,
  )
```

**Extract labels method:**
As we show in `examples/generation.py`, you can create a new task by using our `BaseTask` constructor (line 63). You need to implement the `extract_label` function for your new task. The `extract_label` function should take a string input (LLM generated inference text), and return the label extracted from the input. 

**If no `extract_label` function is provided, the default version will be used, which looks for `final answer:\s+<begin>(.*)<end>` in the LLM generated text.**

**Note: you need to make sure the extracted label are in same format with the `'label'` in your dataset, since the extracted label will be compared with the true label to check correctness of each LLM inference.**

### 4. Further modification of hypogenic

Perhaps your task might have a different way of updating hypotheses, or you want to measure hypotheses' quality slightly differently.

The modularization of the the main classes - generation, update, inference - allows you to inheret from the base class to modify hypogenic's as much as you wish. 

Here are some reasons why you'd modify each class:
**Generation:**  if you want to change the way that hypotheses are initialized, this class, as well as prompt some engineering, would get you there.  Usually, using the default implementation and playing the prompt will suffice.

**Inference:** since predictions based on hypotheses are done in this class, any modification of how predictions are done might require modifying this class.

**Update:**  this class might be the most modifiable, since it contains the entire training loop and certierion for generating new hypotheses.  Here, you can modify the general structure of how and when hypotheses are created.  As a bonus, you can change the **Replace** class in hypogenic/algorithm to modify the behavior of your hypothesis bank.

### 
