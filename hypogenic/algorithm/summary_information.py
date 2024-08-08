import math


class SummaryInformation:
    def __init__(
        self, hypothesis="", acc=1.0, reward=-1, num_visits=1, correct_examples=[]
    ):
        self.hypothesis = hypothesis
        self.acc = acc
        self.num_visits = num_visits
        self.reward = reward
        self.correct_examples = (
            correct_examples  # a list of tuples of the form (sample index, label)
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

    def update_info_if_useful(self, current_example, alpha):
        self.acc = (self.acc * self.num_visits + 1) / (self.num_visits + 1)
        self.num_visits += 1
        self.update_reward(alpha, current_example)

    def update_useful_examples(self, example, label):
        self.correct_examples.append((example, label))

    def update_info_if_not_useful(self, current_example, alpha):
        self.acc = (self.acc * self.num_visits) / (self.num_visits + 1)
        self.num_visits += 1
        self.update_reward(alpha, current_example)

    def __reduce__(self):
        return (self.__class__, (-1, self.acc, self.reward, self.num_visits))

    def __str__(self):
        return f"SI object: reward: {self.reward}, num_visits: {self.num_visits}, correct_examples: {self.correct_examples}"

    def get_examples(self, train_data, task):
        examples = []
        for ex in self.correct_examples:
            index = ex[0]
            label = ex[1]
            examples.append((train_data[PROMPT_NAME_DICT[task]][index], label))
        return examples


PROMPT_NAME_DICT = {
    "shoe": "appearance",
    "hotel_reviews": "review",
    "headline_binary": "headline",
    "retweet": "tweets",
}


def dict_to_summary_information(dict):
    return SummaryInformation(**dict)
