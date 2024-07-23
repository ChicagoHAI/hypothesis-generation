from abc import ABC, abstractmethod
import json
import os
import random
import re

code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")


class Task(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def extract_label(self, text):
        pass

    @abstractmethod
    def get_data(self, num_train, num_test, num_val):
        pass


class Shoe(Task):
    def __init__(self):
        self.task = 'shoe'
        self.label_classes = ['white', 'red', 'orange', 'green', 'blue', 'black']
        self.train_data_path = f'{code_repo_path}/data/shoe_train.json'
        self.test_data_path = f'{code_repo_path}/data/shoe_test.json'
        self.val_data_path = f'{code_repo_path}/data/shoe_val.json'

    def extract_label(self, text):
        if text is None:
            return 'other'

        pattern = r"final answer:\s+(white|red|orange|green|blue|black)"

        match = re.search(pattern, text.lower())
        if match:
            answer = match.group(1)
            if answer in ['white', 'red', 'orange', 'green', 'blue', 'black']:
                return answer
            else:
                return "other"

        return 'other'

    def get_data(self, num_train, num_test, num_val):
        def read_data(file_path, num, is_train=False):
            # Read from json
            with open(file_path, 'r') as f:
                data = json.load(f)
            # shuffle and subsample from data
            if not is_train:
                random.seed(49)

            appearance_all = data['appearance']
            shoe_all = data['shoe']
            label_all = data['label']
            num_samples = min(num, len(data['label']))
            appearance, shoe, label = zip(*random.sample(list(zip(appearance_all, shoe_all, label_all)), num_samples))
            processed_data = {
                'appearance': appearance,
                'shoe': shoe,
                'label': label
            }
            return processed_data

        train_data = read_data(self.train_data_path, num_train, is_train=True)
        test_data = read_data(self.test_data_path, num_test)
        val_data = read_data(self.val_data_path, num_val)

        return train_data, test_data, val_data


class HotelReviews(Task):
    def __init__(self):
        self.task = 'hotel_reviews'

        # The order matters! preceding word should not be substrings of following words
        self.label_classes = ['deceptive', 'truthful']

        self.train_data_path = f'{code_repo_path}/data/hotel_reviews_train.json'
        self.val_data_path = f'{code_repo_path}/data/hotel_reviews_val.json'
        self.test_data_path = f'{code_repo_path}/data/hotel_reviews_test.json'
        self.ood_test_data_path = f'{code_repo_path}/data'

    def extract_label(self, text):
        if text is None:
            return 'other'

        # only keep the part after "Final answer:"
        text = text.lower()
        '''
        if "final answer:" in text:
            text = text[text.index("final answer:") + len("final answer:"):]



        if "label:" in text:
            # only keep the part after "label:"
            text = text[text.index("label:") + len("label:"):]
        '''

        pattern = r"final answer:\s+(truthful|deceptive|other)"

        match = re.search(pattern, text.lower())
        if match:
            answer = match.group(1)
            if answer == "truthful":
                return "truthful"
            elif answer == "deceptive":
                return "deceptive"
            else:
                return "other"

        return 'other'

    def get_data(self, num_train, num_test, num_val):
        def read_data(file_path, num, is_train=False):
            # Read from json
            with open(file_path, 'r') as f:
                data = json.load(f)
            # shuffle and subsample from data
            if not is_train:
                random.seed(49)

            review_all = data['review']
            label_all = data['label']
            num_samples = min(num, len(data['label']))
            review, label = zip(*random.sample(list(zip(review_all, label_all)), num_samples))
            processed_data = {
                'review': review,
                'label': label
            }
            return processed_data

        train_data = read_data(self.train_data_path, num_train, is_train=True)
        test_data = read_data(self.test_data_path, num_test)
        val_data = read_data(self.val_data_path, num_val)

        return train_data, test_data, val_data


class HeadlineBinary(Task):
    def __init__(self):
        self.task = 'headline_binary'
        self.label_classes = ['headline 1', 'headline 2']
        self.train_data_path = f'{code_repo_path}/data/headline_binary_train.json'
        self.test_data_path = f'{code_repo_path}/data/headline_binary_test.json'
        self.val_data_path = f'{code_repo_path}/data/headline_binary_val.json'

    def extract_label(self, text):
        if text is None:
            return 'other'
        text = text.lower()
        pattern = r"answer:\s+(headline 1|headline 2|other)"
        match = re.search(pattern, text.lower())
        '''
        if "Answer:" in text:
            text = text[text.index("Answer:") + len("Answer:"):]
        for x in self.label_classes:
            if x.lower() in text.lower():
                return x
        '''
        if match:
            answer = match.group(1)
            if answer == "headline 1":
                return "headline 1"
            elif answer == "headline 2":
                return "headline 2"
            else:
                return "other"

        return 'other'

    def get_data(self, num_train, num_test, num_val):
        def read_data(file_path, num, is_train=False):
            # Read from json
            with open(file_path, 'r') as f:
                data = json.load(f)
            # shuffle and subsample from data
            if not is_train:
                random.seed(49)

            headline_all = data['headline']
            label_all = data['label']
            num_samples = min(num, len(data['label']))
            headline, label = zip(*random.sample(list(zip(headline_all, label_all)), num_samples))
            processed_data = {
                'headline': headline,
                'label': label
            }
            return processed_data

        train_data = read_data(self.train_data_path, num_train, is_train=True)
        test_data = read_data(self.test_data_path, num_test)
        val_data = read_data(self.val_data_path, num_val)

        return train_data, test_data, val_data


class Retweet(Task):
    def __init__(self):
        self.task = 'retweet'
        self.label_classes = ['first', 'second']
        self.train_data_path = f'{code_repo_path}/data/retweet_train.json'
        self.val_data_path = f'{code_repo_path}/data/retweet_val.json'
        self.test_data_path = f'{code_repo_path}/data/retweet_test.json'

    def extract_label(self, text):
        """
        `text` follows the format "the <label> tweet got more retweets"
        """
        if text is None:
            return 'other'
        text = text.lower()
        import re
        pattern = r"answer: the (\w+) tweet"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return 'other'

    def get_data(self, num_train, num_test, num_val):
        def read_data(file_path, num, is_train=False):
            # Read from json
            with open(file_path, 'r') as f:
                data = json.load(f)
            # shuffle and subsample from data
            if not is_train:
                random.seed(49)

            tweets_all = data['tweets']
            label_all = data['label']
            num_samples = min(num, len(data['label']))
            tweets, label = zip(*random.sample(list(zip(tweets_all, label_all)), num_samples))
            processed_data = {
                'tweets': tweets,
                'label': label
            }
            return processed_data

        train_data = read_data(self.train_data_path, num_train, is_train=True)
        test_data = read_data(self.test_data_path, num_test)
        val_data = read_data(self.val_data_path, num_val)

        return train_data, test_data, val_data


TASKS = {
    'shoe': Shoe,
    'hotel_reviews': HotelReviews,
    'headline_binary': HeadlineBinary,
    'retweet': Retweet,
}
