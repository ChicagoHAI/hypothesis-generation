import pandas as pd
import pickle
import random
import os

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

    def parse_complex_tuple(self, line):
        items = []
        in_quote = False
        buffer = ''
        for char in line:
            if char == "'":
                in_quote = not in_quote
            elif char == ',' and not in_quote:
                items.append(buffer.strip())
                buffer = '' 
            else:
                buffer += char

        if buffer:
            items.append(buffer.strip())

        return tuple(item.strip("'") for item in items)

    def read_data(self, file_path, num_examples):
        data = []

        with open(file_path, 'r') as file:
            for line in file:
                clean_line = line.strip().strip('()')
                tuple_data = self.parse_complex_tuple(clean_line)
                data.append(tuple_data)
        if self.is_train:
            random.shuffle(data)
        
        data = self.preprocess_data(data, num_examples)

        output = {}
        for key in data:
            output[key] = data[key][:num_examples]
        
        return output

        

    # def read_data(self):
    #     """
    #     Read data from file_path and return train and test data.

    #     data format:
    #     [
    #         (
    #             'a young and tall man with blue hat, black shirt, and a small white bag',
    #             'the color of shoes is black'
    #         ),
    #         ...
    #     ]
    #     """
    #     examples = pickle.load(open(file_path, 'rb'))

    #     train_data = examples[:1000]
    #     test_data = examples[1000:]

    #     # shuffle data
    #     random.shuffle(train_data)
    #     random.shuffle(test_data)

    #     # sample 20 test examples
    #     train_data = train_data[:num_train]
    #     test_data = test_data[:num_test]
        
    #     return train_data, test_data

    def preprocess_data(self, data, num_examples):
        # takes in output from read_data, turn into a dictionary
        output = {}
        output['appearance'] = [item[0] for item in data[:num_examples]]
        output['shoe'] = [item[1] for item in data[:num_examples]]
        output['label'] = [item[1][len('the color of shoe is ')+1:] for item in data[:num_examples]]
        return output

    def get_data(self):
        return self.data


class HotelReviewsDataProcessor:
    def __init__(self, file_path, num_examples, is_train=False):
        self.task_name = 'hotel_reviews'
        self.file_path = file_path
        self.num_examples = num_examples
        self.is_train = is_train
        df = self.read_data()
        self.data = self.preprocess_data(df)

    def read_data(self):
        df = pd.read_json(self.file_path, lines=True)
        return df
    
    def dataframe_to_dict(self, dataframe):
        """
        dataframe: A dataframe containing reviews for hotels. Each row is a hotel, and the columns are different reviews for that hotel.
        returns: A dictionary keys 'reviews' and 'labels'. The value of 'reviews' is a list of reviews, and the value of 'labels' is a list of labels.
        """
        output = {}

        if self.num_examples <= 0:
            return output
        
        positive_truthful_reviews = dataframe['positive_truthful'].tolist()
        positive_deceptive_reviews = dataframe['positive_deceptive'].tolist()
        negative_truthful_reviews = dataframe['negative_truthful'].tolist()
        negative_deceptive_reviews = dataframe['negative_deceptive'].tolist()

        reviews = positive_truthful_reviews + positive_deceptive_reviews + negative_truthful_reviews + negative_deceptive_reviews
        labels = ['truthful'] * len(positive_truthful_reviews) + ['deceptive'] * len(positive_deceptive_reviews) + ['truthful'] * len(negative_truthful_reviews) + ['deceptive'] * len(negative_deceptive_reviews)

        if self.is_train:
            if self.num_examples <= len(labels):
                reviews, labels = zip(*random.sample(list(zip(reviews, labels)), self.num_examples))
        elif self.num_examples <= len(labels):
            random.seed(49)
            reviews, labels = zip(*random.sample(list(zip(reviews, labels)), self.num_examples))
        else:
            num_example_per_type = self.num_examples // 4
            extra_examples = self.num_examples % 4
            reviews = positive_truthful_reviews[:num_example_per_type] + positive_deceptive_reviews[:num_example_per_type] + negative_truthful_reviews[:num_example_per_type] + negative_deceptive_reviews[:num_example_per_type]
            labels = ['truthful'] * num_example_per_type + ['deceptive'] * num_example_per_type + ['truthful'] * num_example_per_type + ['deceptive'] * num_example_per_type
            if extra_examples == 1:
                reviews.append(positive_truthful_reviews[num_example_per_type])
                labels.append('truthful')
            elif extra_examples == 2:
                reviews.append(positive_truthful_reviews[num_example_per_type])
                labels.append('truthful')
                reviews.append(positive_deceptive_reviews[num_example_per_type])
                labels.append('deceptive')
            elif extra_examples == 3:
                reviews.append(positive_truthful_reviews[num_example_per_type])
                labels.append('truthful')
                reviews.append(positive_deceptive_reviews[num_example_per_type])
                labels.append('deceptive')
                reviews.append(negative_truthful_reviews[num_example_per_type])
                labels.append('truthful')

        output['review'] = reviews
        output['label'] = labels

        return output
    
    def preprocess_data(self, data):
        """
        Reads in the selected training examples in a dataframe format and turn it into a dictionary.

        Input:
        - data: a dataframe containing the selected training examples

        Output:
        - output: a dictionary containing the selected training examples
        """

        output = self.dataframe_to_dict(data)

        return output
    
    def get_data(self):
        return self.data


class HeadlineProcessor:
    def __init__(self, file_path, num, is_train=False):
        self.task_name = 'headline'
        self.file_path = file_path
        self.is_train = is_train
        self.data = self.read_data(file_path, num)

    def read_data(self, file_path, num):
        df = pd.read_csv(file_path)

        unique_ids = list(df['clickability_test_id'].unique())
        if self.is_train:
            random.shuffle(unique_ids)
        data = self.preprocess_data(unique_ids[:num], df)
        
        return data

    def preprocess_data(self, unique_ids, df):
        # takes in output from read_data, turn into a dictionary
        output = {}
        output['headline'] = []
        output['label'] = []

        if not self.is_train:
            random.seed(49)
        for _, uid in enumerate(unique_ids):
            df_filtered = df[df['clickability_test_id'] == uid]['headline'].to_list()
            df_filtered_list = list(enumerate(df_filtered))
            random.shuffle(df_filtered_list)
            headline = [row[1] for row in df_filtered_list]
            label_index = [row[0] for row in df_filtered_list]
            output['headline'].append(headline)
            if label_index[0] == 0:
                label = "headline 1"
            else:
                label = "headline 2"
            output['label'].append(label)
        return output
    
    def get_data(self):
        return self.data


class RetweetProcessor:
    def __init__(self, file_path, num_examples, is_train=False):
        self.task_name = 'retweet'
        self.file_path = file_path
        self.is_train = is_train
        self.data = self.read_data(file_path, num_examples)

    def read_data(self, file_path, num_examples):
        df = pd.read_csv(file_path)

        if self.is_train:
            # shuffle the rows in the dataframe
            df = df.sample(frac=1).reset_index(drop=True)
        
        # dataset contains keys "first_text", "second_text", "label"
        dataset = {}
        dataset['first_text'] = df['first_text'].tolist()
        dataset['second_text'] = df['second_text'].tolist()
        # label is "first" if first_retweet > second_retweet, "second" if second_retweet > first_retweet, "same" if equal
        dataset['label'] = ['first' if a > b else 'second' if b > a else 'same' for a, b in zip(df['first_retweet'], df['second_retweet'])]

        assert dataset['label'].count('same') == 0, "There should be no 'same' labels in the dataset"

        data = {}
        data['first_text'] = dataset['first_text'][:num_examples]
        data['second_text'] = dataset['second_text'][:num_examples]
        data['label'] = dataset['label'][:num_examples]
        
        # get rid of preceding and trailing whitespaces
        data['first_text'] = [item.strip() for item in data['first_text']]
        data['second_text'] = [item.strip() for item in data['second_text']]

        # reform data
        data = self.reform_data(data)

        return data
    
    def reform_data(self,data):
        reformed_data = {}
        tweets_list = []
        for i in range(len(data['label'])):
            tweets_list.append([data['first_text'][i],data['second_text'][i]])
        reformed_data['tweets'] = tweets_list
        reformed_data['label'] = data['label']
        
        return reformed_data
    
    def get_data(self):
        return self.data


DATA_PROCESSORS = {
    'shoe': ShoeRecommendationDataProcessor,
    'hotel_reviews': HotelReviewsDataProcessor,
    'headline_binary': HeadlineProcessor,
    'retweet': RetweetProcessor,
}