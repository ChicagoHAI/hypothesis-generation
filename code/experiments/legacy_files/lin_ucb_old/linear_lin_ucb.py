import matplotlib.pyplot as plt
import random
import sys

sys.path.append('../')
from utils import prep_shoe_data
from data_loader import get_data

from lin_ucb.lin_ucb import LinearUCB
from lin_ucb.bandit import ContextualBandit
from lin_ucb.dataset.ucb_dataset import ShoeDataset

def train(seed, alpha, num_train, task, num_test, num_arms, num_features, evaluate_steps):
    """
    train_acc: a list of train accuracies at each evaluation step
    test_acc: a list of test accuracies at each evaluation step
    """
    # Training prep
    train_data, test_data = get_data(task, num_train, num_test)
    train_x, train_y = prep_shoe_data(train_data)
    train_dataset = ShoeDataset(x=train_x, y=train_y)
    train_bandit = ContextualBandit(T=num_train, n_arms=num_arms, seed=seed, dataset=train_dataset)

    # Training
    linucb = LinearUCB(bandit=train_bandit, num_features=num_features, alpha=alpha)
    linucb.train()
    train_acc = round(1-sum(linucb.regrets)/len(linucb.regrets), 3)

    # Testing prep
    test_x, test_y = prep_shoe_data(test_data)
    test_dataset = ShoeDataset(x=test_x, y=test_y)
    test_bandit = ContextualBandit(T=num_test, n_arms=num_arms, seed=seed, dataset=test_dataset)

    # Testing
    linucb.test(test_bandit)
    test_acc = round(1-sum(linucb.regrets)/len(linucb.regrets), 3)

    return train_acc, test_acc


def lin_ucb(seed, alpha, num_train, task='shoe', num_test=100, num_arms=6, num_features=7, evaluate_steps=100):
    # Set up
    seed = seed
    alpha = alpha
    random.seed(seed)

    train_acc_dict = {}
    test_acc_dict = {}

    for num_train in [100, 200, 500, 1000]:
        train_acc, test_acc = train(seed, alpha, num_train, task, num_test, num_arms, num_features, evaluate_steps)
        train_acc_dict[num_train] = train_acc
        test_acc_dict[num_train] = test_acc

    return train_acc_dict, test_acc_dict


def main():
    seeds = [0, 1, 2, 3, 4]
    alphas = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
    
    train_acc_by_alpha = {}
    test_acc_by_alpha = {}
    for alpha in alphas:
        for seed in seeds:
            train_acc_dict, test_acc_dict = lin_ucb(seed, alpha, num_train=100)
            if alpha in train_acc_by_alpha:
                train_acc_by_alpha[alpha].append(train_acc_dict)
            else:
                train_acc_by_alpha[alpha] = [train_acc_dict]
            if alpha in test_acc_by_alpha:
                test_acc_by_alpha[alpha].append(test_acc_dict)
            else:
                test_acc_by_alpha[alpha] = [test_acc_dict]

    print(f'train_acc_by_alpha: {train_acc_by_alpha}')
    print(f'test_acc_by_alpha: {test_acc_by_alpha}')

    # save to pickle
    import pickle
    with open('./train_acc_by_alpha.pkl', 'wb') as f:
        pickle.dump(train_acc_by_alpha, f)
    with open('./test_acc_by_alpha.pkl', 'wb') as f:
        pickle.dump(test_acc_by_alpha, f)

if __name__ == '__main__':
    main()
