import os
import sys

code_repo_path = os.environ.get("CODE_REPO_PATH")
sys.path.append(f'{code_repo_path}/code/')

import utils
import pickle

file = "/data/tejess/hypothesis_generation/results/claude_exp/hypotheses_final.pkl"
with open(file, 'rb') as f:
    d = pickle.load(f)
    for x in d:
        print("Hypothesis:", x, "\nReward:", d[x].reward, "Accuracy:", d[x].acc, "Num Examples visited by:", d[x].num_visits)