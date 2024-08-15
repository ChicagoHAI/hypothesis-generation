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
```bash
hypogenic_generation --args
```
For help with the arguments, run:
```bash
hypogenic_generation --help
```
### 3. Hypothesis Inference
```bash
hypogenic_inference --args
```
For help with the arguments, run:
```bash
hypogenic_inference --help
```
**We will support command lines for HypoGeniC on new tasks and datasets in a later release.**

## Use HypoGeniC in your code

In order to use hypogenic with your own data and in you own code, follow the following steps:
