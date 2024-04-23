# adapted from: https://github.com/sauxpa/neural_exploration

import numpy as np
import random
import torch

class ContextualBandit():
    def __init__(self,
                 T,
                 n_arms,
                 seed=None,
                 dataset=None,
                 ):
        # if not None, freeze seed for reproducibility
        self._seed(seed)

        # number of rounds
        self.T = T
        # number of arms
        self.n_arms = n_arms

        self.dataset = dataset

        # (T=2000, n_arms=2, sentence)
        self.features = dataset.get_features()
        # (T=2000, n_arms=2)
        self.rewards = dataset.get_rewards()

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
