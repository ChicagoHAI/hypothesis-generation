# LLM Hypothesis Generation to Improve Inductive Reasoning Abilities

To mimic human cognitive process in inductive reasoning, we prompt LLM to generate hypothesis while performing online learning.

Our method is able to
* overcome context window size limit,
* provide interpretability,
* and improve model ability across various tasks.

## Install environment
The environment configuration file is located at conda_env_configs/hypogenic.yml
```
conda env create -f hypogenic.yml
```

## Set up path

```bash
export PATH_PREFIX=<parent path to this directory>
export CODE_REPO_PATH=<path to this directory>
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
1. [Optional] Start Redis server
```bash
cd $PATH_PREFIX/redis-stable/src
./redis-server
```
2. Hypothesis generation
```
python $CODE_REPO_PATH/algorithm/algorithm_generation.py  
    --seeds SEEDS --task TASK --model MODEL 
    --generation_style GENERATION_STYLE
    --inference_style INFERENCE_STYLE
    --update_style UPDATE_STYLE
    --replace_style REPLACE_STYLE
    [--model_path MODLE_PATH] 
    [--use_cache USE_CACHE]
    [--verbose] [--use_system_prompt] 
    [--num_init NUM_INIT]
    [--init_hypotheses_per_batch INIT_HYPOTHESES_PER_BATCH]
    [--init_batch_size INIT_BATCH_SIZE] 
    [--num_train NUM_TRAIN] 
    [--num_test NUM_TEST]
    [--num_val NUM_VAL]
    [--save_every_n_examples SAVE_EVERY_N_EXAMPLES]
    [--k K] [--max_num_hypotheses MAX_NUM_HYPOTHESES]
    [--num_hypotheses_to_update NUM_HYPOTHESES_TO_UPDATE]
    [--update_batch_size UPDATE_BATCH_SIZE]
    [--alpha ALPHA] [--num_wrong_scale NUM_WRONG_SCALE]
    [--old_hypothesis_file OLD_HYPOTHESIS_FILE]
    [--sample_num_to_restart_from SAMPLE_NUM_TO_RESTART_FROM]
    [--epoch_to_restart_from EPOCH_TO_RESTART_FROM]
    [--num_epochs NUM_EPOCHS]
    [--output_folder OUTPUT_FOLDER]

required arguments: 
    --seeds SEEDS
        Set random seeds. If a list of seeds is passed, the 
        program will run aloop for all seeds.
    --task TASK
        Set task to run. 
        Options: [shoe,hotel_reviews,
                  headline_binary,retweet]
    -- model MODEL
        Set model to run.
        Options: Please see consts/model_consts.py
    
                                
```
<!-- ## Add your own dataset
1. Add your dataset to `data` directory.
2. Update code accordingly:
    * `code/tasks.py`: define the new task.
    * `code/data_loader.py`: add the data path to load the new data.
    * `code/data_processor.py`: write processor for the new dataset.
    * `code/prompt.py`: add prompt for the new dataset. -->
