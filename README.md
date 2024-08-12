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
    - `<TASK>_train.json`: A json file containing the training data. The json file should have keys: `'text_features_1'`, ... `'text_features_n'`, `'label'`. The values corresponding to each key should be a list of strings.
    



