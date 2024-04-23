import os
import sys

code_repo_path = os.environ.get("CODE_REPO_PATH")
sys.path.append(f'{code_repo_path}/code/')

import utils
import pickle

file="/data/tejess/hypothesis_generation/results/claude_exp/49_hypothesesinit_test100_strategybest_few_shot0_results.pkl"
with open(file, 'rb') as f:
    d = pickle.load(f)
    print(d)