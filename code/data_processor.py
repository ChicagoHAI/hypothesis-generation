import pandas as pd
import pickle
import random
import os
import json

from datasets import load_dataset
from tasks import TASKS


code_repo_path = os.environ.get("CODE_REPO_PATH")

if code_repo_path:
    print(f"Code repo path: {code_repo_path}")
else:
    print("Environment variable not set.")


class ShoeRecommendationDataProcessor:
    def __init__(self, file_path, num, is_train=False):
        self.task_name = 'shoe'
        self.file_path = file_path
        self.is_train = is_train
        self.data = self.read_data(file_path, num)

    def read_data(self, file_path, num):
        # Read from json
        with open(file_path, 'r') as f:
            data = json.load(f)
        # shuffle and subsample from data
        if not self.is_train:
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
    
    def get_data(self):
        return self.data


class HotelReviewsDataProcessor:
    def __init__(self, file_path, num, is_train=False):
        self.task_name = 'hotel_reviews'
        self.file_path = file_path
        self.is_train = is_train
        self.data = self.read_data(file_path, num)

    def read_data(self, file_path, num):
        # Read from json
        with open(file_path, 'r') as f:
            data = json.load(f)
        # shuffle and subsample from data
        if not self.is_train:
            random.seed(49)

        review_all = data['review']
        label_all = data['label']
        num_samples = min(num,len(data['label']))
        review, label = zip(*random.sample(list(zip(review_all, label_all)), num_samples))
        processed_data = {
            'review': review,
            'label': label
        }
        return processed_data
    
    def get_data(self):
        return self.data


class HeadlineProcessor:
    def __init__(self, file_path, num, is_train=False):
        self.task_name = 'headline'
        self.file_path = file_path
        self.is_train = is_train
        self.data = self.read_data(file_path, num)

    def read_data(self, file_path, num):
        # Read from json
        with open(file_path, 'r') as f:
            data = json.load(f)
        # shuffle and subsample from data
        if not self.is_train:
            random.seed(49)

        headline_all = data['headline']
        label_all = data['label']
        num_samples = min(num,len(data['label']))
        headline, label = zip(*random.sample(list(zip(headline_all, label_all)), num_samples))
        processed_data = {
            'headline': headline,
            'label': label
        }
        return processed_data
    
    def get_data(self):
        return self.data


class RetweetProcessor:
    def __init__(self, file_path, num, is_train=False):
        self.task_name = 'retweet'
        self.file_path = file_path
        self.is_train = is_train
        self.data = self.read_data(file_path, num)

    def read_data(self, file_path, num):
        # Read from json
        with open(file_path, 'r') as f:
            data = json.load(f)
        # shuffle and subsample from data
        if not self.is_train:
            random.seed(49)

        tweets_all = data['tweets']
        label_all = data['label']
        num_samples = min(num,len(data['label']))
        tweets, label = zip(*random.sample(list(zip(tweets_all, label_all)), num_samples))
        processed_data = {
            'tweets': tweets,
            'label': label
        }
        return processed_data
    
    def get_data(self):
        return self.data


DATA_PROCESSORS = {
    'shoe': ShoeRecommendationDataProcessor,
    'hotel_reviews': HotelReviewsDataProcessor,
    'headline_binary': HeadlineProcessor,
    'retweet': RetweetProcessor,
}