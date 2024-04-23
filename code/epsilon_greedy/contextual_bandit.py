class ContextualBandit:
    """
    Bandit with a context-dependent reward function.

    For our shoe sales example, the context is the user's profile, and the reward is the probability that the user will buy the recommended shoe.

    In other words, this is the environment that the agent interacts with, and it is constructed based on the dataset.
    """
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
    
    def get_reward(self, action, context):
        # get reward based on prediction accuracy and add noise

        # action (prediction of shoe color) is correct only if it matches the shirt color 
        true_action_value = 1 if context[4] == action else 0

        # noise = np.random.normal(0, 0.1)
        noise = 0

        return true_action_value + noise

