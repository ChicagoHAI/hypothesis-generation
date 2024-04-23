# process data for training (huggingface compatible)
import numpy as np
import torch

class ShoeDataset():
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index_tuple):
        # index_tuple = (time step, arm)
        # Note: the feature for any arm is the same
        # print(f'self.get_feature(index_tuple[0]): {self.get_feature(index_tuple[0])}')
        return self.get_feature(index_tuple[0])

    def get_feature(self, idx):
        return {
            'x': self.x[idx],
            'labels': self.get_label(idx)
        }
    
    def get_label(self, idx):
        # return one hot encoding of label
        # y: (n,)
        # label: (6,)
        label = np.zeros(6)
        label[int(self.y[idx])] = 1
        return label
    
    def get_features(self):
        return self.x

    def get_rewards(self):
        # one hot encoders of y
        # y: (n,)
        # rewards: (n, 6)
        rewards = np.zeros((self.y.shape[0], 6))
        for i in range(self.y.shape[0]):
            rewards[i][int(self.y[i])] = 1
        return rewards
    