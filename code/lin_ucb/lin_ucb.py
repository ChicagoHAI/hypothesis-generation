import numpy as np
from tqdm import tqdm

class LinearUCB:
    def __init__(self, bandit, num_features, alpha=1.0):
        self.bandit = bandit  # The contextual bandit to solve

        self.num_features = num_features  # Number of features in the linear model
        self.alpha = alpha  # Exploration parameter

        # Initialize model parameters and covariance matrix
        self.theta = np.zeros((bandit.n_arms, num_features))
        self.A = np.array([np.identity(num_features) for _ in range(bandit.n_arms)])
        self.b = np.zeros((bandit.n_arms, num_features))

        # Initialize logging
        self.action = -1
        self.ucb_scores = np.zeros(bandit.n_arms)
        self.actions = np.empty(self.bandit.T).astype('int')
        self.regrets = np.empty(self.bandit.T)
        self.iteration = 0
        self.throttle = int(1e2)

    def select_action(self, context):
        # Compute the UCB scores for each action
        self.ucb_scores = np.zeros(self.bandit.n_arms)
        for action in range(self.bandit.n_arms):
            self.ucb_scores[action] = self.compute_ucb_scores(context, action)

        # Choose the action with the highest UCB score
        self.action = np.argmax(self.ucb_scores)
        # print(f'action: {self.action}')

    def compute_ucb_scores(self, context, action):
        theta_hat = np.linalg.solve(self.A[action], self.b[action])
        ucb_score = np.dot(context, theta_hat) + \
                          self.alpha * \
                          np.sqrt(np.dot(context, np.linalg.solve(self.A[action], context)))
        # print(f'ucb_score: {ucb_score}')
        return ucb_score
    
    def update(self, action, context, reward):
        # Update the model parameters based on the chosen action and observed reward
        self.A[action] = self.A[action] + np.outer(context, context)
        self.b[action] = self.b[action] + reward * context
        self.theta[action] = np.linalg.solve(self.A[action], self.b[action])

    def train(self):
        postfix = {
            'total regret': 0.0,
            '% optimal arm': 0.0,
        }
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):
                context = self.bandit.features[t]
                context = np.array(context)
                # print(f'context: {context}')
                self.select_action(context)
                reward = self.bandit.rewards[t, self.action]
                # print(f'reward: {reward}')
                self.update(self.action, context, reward)
                # print(f'theta: {self.theta}')

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

    def test(self, test_bandit):
        # update bandit to test bandit
        self.bandit = test_bandit
        self.T = self.bandit.T
        self.iteration = 0
        self.actions = np.empty(self.bandit.T).astype('int')
        self.regrets = np.empty(self.bandit.T)

        postfix = {
            'total regret': 0.0,
            '% optimal arm': 0.0,
        }
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):
                context = self.bandit.features[t]
                context = np.array(context)
                self.select_action(context)

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