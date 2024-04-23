# LLM Hypothesis Generation to Improve Inductive Reasoning Abilities

To mimic human cognitive process in inductive reasoning, we prompt LLM to generate hypothesis while performing online learning.

Our method is able to
* overcome context window size limit,
* provide interpretability,
* and improve model ability across various tasks.

## Install environment
```
conda env create -f inductive_reasoning.yml
```

## Set up path
```
export CODE_REPO_PATH=<path to this directory>
```

## Add your own dataset
1. Add your dataset to `data` directory.
2. Update code accordingly:
    * `code/tasks.py`: define the new task.
    * `code/data_loader.py`: add the data path to load the new data.
    * `code/data_processor.py`: write processor for the new dataset.
    * `code/prompt.py`: add prompt for the new dataset.
