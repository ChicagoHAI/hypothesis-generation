# adapted from: https://github.com/sauxpa/neural_exploration

import numpy as np
import abc
import torch
import torch.nn as nn

from tqdm import tqdm

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv


class UCB(abc.ABC):
    """Base class for UBC methods.
    """
    def __init__(self,
                 bandit,
                 reg_factor=1.0,
                 confidence_scaling_factor=-1.0,
                 delta=0.1,
                 train_every=1,
                 throttle=int(1e2),
                 exploration_params=None,
                 ):
        # bandit object, contains features and generated rewards
        self.bandit = bandit
        # L2 regularization strength
        self.reg_factor = reg_factor
        # Confidence bound with probability 1-delta
        self.delta = delta
        # multiplier for the confidence bound 
        self.confidence_scaling_factor = confidence_scaling_factor

        # train approximator only every few rounds
        self.train_every = train_every

        # throttle tqdm updates
        self.throttle = throttle

        if self.exploration_params is None:
            self.exploration_params = [0.1 for _ in range(self.bandit.T)]

        self.reset()

    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        self.exploration_bonus = np.empty((self.bandit.T, self.bandit.n_arms))
        self.mu_hat = np.empty((self.bandit.T, self.bandit.n_arms))
        self.upper_confidence_bounds = np.ones((self.bandit.T, self.bandit.n_arms))

    def reset_regrets(self):
        """Initialize regrets.
        """
        self.regrets = np.empty(self.bandit.T)

    def reset_actions(self):
        """Initialize cache of actions.
        """
        self.actions = np.empty(self.bandit.T).astype('int')

    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = np.array(
            [
                np.eye(self.approximator_dim)/self.reg_factor for _ in self.bandit.arms
            ]
        )

        # print('A_inv dim: ', self.A_inv.shape)

    def reset_b(self):
        """Initialize n_arm vectors representing the exploration bonus bias terms.
        """
        self.b = np.array(
            [
                np.zeros(self.approximator_dim) for _ in self.bandit.arms
            ]
        )

        # print('b dim: ', self.b.shape)

    def sample_action(self):
        """Return the action to play based on current estimates
        """
        return np.argmax(self.upper_confidence_bounds[self.iteration]).astype('int')

    @abc.abstractmethod
    def reset(self):
        """Initialize variables of interest.
        To be defined in children classes.
        """
        pass

    @property
    @abc.abstractmethod
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        pass

    @abc.abstractmethod
    def train(self):
        """Update approximator.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass

    def update_confidence_bounds(self):
        """Update confidence bounds and related quantities for all arms.
        """

        # print(f'self.bandit.arms: {self.bandit.arms}')
        # print(f'self.bandit.features[self.iteration]: {self.bandit.features[self.iteration]}')

        # UCB exploration bonus
        self.exploration_bonus[self.iteration] = np.array(
            [
                # alpha_t * sqrt(x_t.T A_t^{-1} x_t)
                self.exploration_params[self.iteration] * np.sqrt(np.dot(self.model.last_hidden_output(self.bandit.features[self.iteration].to(self.device)).detach().cpu().numpy(), 
                                                                         np.dot(self.A_inv[a], 
                                                                                self.model.last_hidden_output(self.bandit.features[self.iteration].to(self.device)).detach().cpu().numpy()
                                                                                )
                                                                        )
                                                                ) for a in self.bandit.arms
            ]
        )

        # update reward prediction mu_hat
        self.predict()

        # estimated combined bound for reward
        self.upper_confidence_bounds[self.iteration] = self.mu_hat[self.iteration] + self.exploration_bonus[self.iteration]

    def update_A_inv(self):
        # print('Update A_inv')
        # print('A_inv dim: ', self.A_inv.shape)
        for action in range(len(self.bandit.arms)):
            prev = self.A_inv[action]
            self.A_inv[action] = inv_sherman_morrison(
                self.model.last_hidden_output(
                    self.bandit.features[self.iteration].to(self.device)
                ).detach().cpu().numpy(),
                self.A_inv[action]
            )
        # print('A_inv dim: ', self.A_inv.shape)
            

    def update_b(self):
        # print('Update b')
        # print('b dim: ', self.b.shape)
        for action in range(len(self.bandit.arms)):
            prev = self.b[action]
            self.b[action] = self.b[action] + self.model.last_hidden_output(
                self.bandit.features[self.iteration].to(self.device)
            ).detach().cpu().numpy() * self.bandit.rewards[self.iteration][action]
        # print('b dim: ', self.b.shape)

    def update_theta(self):
        # print('A_inv[action] dim: ', self.A_inv[self.action].shape)
        # print('b[action] dim: ', self.b[self.action].shape)
        # print('theta dim (before):', self.model.theta.weight.shape)
        # print('theta dim (after):', np.dot(self.A_inv[self.action], self.b[self.action]).shape)
        # print('theta weights (before):', self.model.theta.weight)
        prev = self.model.theta.weight

        # A_inv: (6, 100, 100)
        # b: (6, 100)
        # theta = A_inv * b (shape = (6, 100))
        theta = np.zeros((self.A_inv.shape[0], self.A_inv.shape[1]))
        for i in range(self.A_inv.shape[0]):
            theta[i] = np.dot(self.A_inv[i], self.b[i])
        # print('theta dim (after):', theta.shape)
        self.model.theta.weight = nn.Parameter(torch.Tensor(theta).float().to(self.device))

        # print('theta weights (after):', self.model.theta.weight)
        if torch.all(torch.eq(prev, self.model.theta.weight)):
            print('No update to theta')

    def run(self):
        """Run an episode of bandit.
        """
        postfix = {
            'total regret': 0.0,
            '% optimal arm': 0.0,
        }
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):
                # update confidence of all arms based on observed features at time t
                self.update_confidence_bounds()
                # pick action with the highest boosted estimated reward
                self.action = self.sample_action()
                self.actions[t] = self.action
                # update approximator
                if t % self.train_every == 0:
                    print('Training...')
                    self.train()
                # update exploration indicator A_inv
                self.update_A_inv()
                self.update_b()
                self.update_theta()
                # compute regret
                self.regrets[t] = self.bandit.best_rewards_oracle[t]-self.bandit.rewards[t][self.action]
                # increment counter
                self.iteration += 1

                # log
                postfix['total regret'] += self.regrets[t]
                n_optimal_arm = np.sum(
                    self.actions[:self.iteration] == self.bandit.best_actions_oracle[:self.iteration]
                )
                postfix['% optimal arm'] = '{:.2%}'.format(n_optimal_arm / self.iteration)

                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)

    def run_inference(self):
        postfix = {
            'total regret': 0.0,
            '% optimal arm': 0.0,
        }
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):
                
                # pick action with the highest boosted estimated reward
                self.action = self.sample_action()
                self.actions[t] = self.action
               
                # compute regret
                self.regrets[t] = self.bandit.best_rewards_oracle[t]-self.bandit.rewards[t][self.action]
                # increment counter
                self.iteration += 1

                # log
                postfix['total regret'] += self.regrets[t]
                n_optimal_arm = np.sum(
                    self.actions[:self.iteration] == self.bandit.best_actions_oracle[:self.iteration]
                )
                postfix['% optimal arm'] = '{:.2%}'.format(n_optimal_arm / self.iteration)

                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)