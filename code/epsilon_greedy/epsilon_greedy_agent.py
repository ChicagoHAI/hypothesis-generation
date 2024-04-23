import numpy as np
from .contextual_bandit import ContextualBandit

class EpsilonGreedyAgent:
    """
    Epsilon greedy agent for contextual bandits.

    This is the "model" that we train.
    """
    def __init__(self, n_actions, n_features, epsilon=0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.epsilon = epsilon
        self.q_values = np.zeros((n_actions, n_features))
        self.action_counts = np.zeros(n_actions) 
    
    def select_action(self, context):
        if np.random.rand() < self.epsilon:
            # Explore: Choose a random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: Choose the action with the highest estimated value
            action_values = np.dot(self.q_values, context)
            return np.argmax(action_values)
    
    def update_q_values(self, action, context, reward):
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action] @ context)
        print('self.q_values: ', self.q_values)

def run_epislon_greedy(np_seed, train_steps, train_x, test_x):
    np.random.seed(np_seed)

    # Define the number of actions and features
    n_actions = 6 # 6 colors
    n_features = 7 # 7 words in the description

    # Create a contextual bandit environment
    bandit = ContextualBandit(n_actions, n_features)
    # Create an agent
    agent = EpsilonGreedyAgent(n_actions, n_features, epsilon=0.1)

    # Simulation loop
    n_steps = train_steps
    total_rewards = 0

    # Training: update q values
    for step in range(n_steps):
        context = train_x[step]  # Get the context from the environment
        action = agent.select_action(context)  # Choose an action
        reward = bandit.get_reward(action, context)  # Observe the reward
        agent.update_q_values(action, context, reward)  # Update the Q-values
        total_rewards += reward

    print("(Train) Total rewards:", total_rewards)

    # Testing: evaluate the agent based on regret
    n_steps = 100
    regret = 0
    for step in range(n_steps):
        context = test_x[step]  # Get the context from the environment
        action = agent.select_action(context)  # Choose an action
        reward = bandit.get_reward(action, context)  # Observe the reward
        regret += (1 - reward)

    print("(Test) Regret:", regret)