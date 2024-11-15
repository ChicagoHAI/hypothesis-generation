import math

import pandas as pd


class SummaryInformation:
    """
    This class is meant to keep track of how well a hypothesis performs
    """

    def __init__(
        self, hypothesis="", acc=0.0, reward=0, num_visits=0, correct_examples=None
    ):
        """
        Initialize the SummaryInformation object

        Parameters:
            hypothesis: the hypothesis that the object is tracking
            acc: the accuracy of the hypothesis
            reward: the reward of the hypothesis
            num_visits: the number of times the hypothesis has been visited
            correct_examples: a list of tuples of the form (sample index, label)
        """
        self.hypothesis = hypothesis
        self.acc = acc  # accuracy
        self.num_visits = num_visits
        self.reward = reward
        self.correct_examples = (
            correct_examples  # a list of tuples of the form (sample index, label)
            if correct_examples is not None
            else []
        )

    def set_accuracy(self, new_accuracy):
        self.acc = new_accuracy

    def set_num_visits(self, new_num_visits):
        self.num_visits = new_num_visits

    def set_reward(self, new_reward):
        self.reward = new_reward

    def set_example(self, new_examples):
        self.correct_examples = new_examples

    def set_hypothesis(self, hypothesis):
        self.hypothesis = hypothesis

    def update_reward(self, alpha, num_examples):
        self.reward = self.acc + alpha * math.sqrt(
            math.log(num_examples) / self.num_visits
        )

    # In the update function, if it got the ith sample correct, ajust accuracy accordingly
    def update_info_if_useful(self, current_example, alpha):
        self.acc = (self.acc * self.num_visits + 1) / (self.num_visits + 1)
        self.num_visits += 1
        self.update_reward(alpha, current_example)

    def update_useful_examples(self, example, label):
        self.correct_examples.append((example, label))

    # In the update function, if it got the ith smample wrong, adjust accuracy accordinly
    def update_info_if_not_useful(self, current_example, alpha):
        self.acc = (self.acc * self.num_visits) / (self.num_visits + 1)
        self.num_visits += 1
        self.update_reward(alpha, current_example)

    def __reduce__(self):
        return (self.__class__, (-1, self.acc, self.reward, self.num_visits))

    def __str__(self):
        return f"SI object: reward: {self.reward}, num_visits: {self.num_visits}, correct_examples: {self.correct_examples}"

    # essentially redistributes the dataset
    def get_examples(self, train_data: pd.DataFrame):
        return train_data.iloc[[index for index, *_ in self.correct_examples]]

    @staticmethod
    def from_dict(data: dict) -> "SummaryInformation":
        return SummaryInformation(**data)
