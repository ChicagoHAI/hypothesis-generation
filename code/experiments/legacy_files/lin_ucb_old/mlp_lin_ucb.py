import matplotlib.pyplot as plt
import random
from neural_lin_ucb.neural_lin_ucb import NeuralLinUCB
from neural_lin_ucb.bandit import ContextualBandit
from neural_lin_ucb.shoe_data_processor import ShoeDataset
from utils import prep_shoe_data
from data_loader import get_data


def build_neural_lin_ucb(train_x, train_y, steps, seed, n_arms, max_steps):
    dataset = ShoeDataset(x=train_x, y=train_y)
    bandit = ContextualBandit(T=steps, n_arms=n_arms, seed=seed, dataset=dataset)
    neurallinucb = NeuralLinUCB(bandit=bandit, max_steps=max_steps)

    return neurallinucb


def log_regret_info(neurallinucb, msg):
    # plot the regret
    print(f'{msg} regrets: ', neurallinucb.regrets)
    average_regret = [sum(neurallinucb.regrets[:i])/i for i in range(1, len(neurallinucb.regrets))]
    print(f'{msg} average_regret: ', average_regret)
    # plot average regret
    plt.plot(average_regret, label=f'{msg} average regret')
    # save plots
    plt.savefig(f'./{msg}_average_regret.png')


def main():

    msg='train_200_epoch_1'

    seed=42
    n_arms=6
    task = 'shoe'
    num_train = 200
    num_test = 100
    max_steps=0

    random.seed(seed)
    train_data, test_data = get_data(task, num_train, num_test)
    
    # training
    train_x, train_y = prep_shoe_data(train_data)
    neurallinucb = build_neural_lin_ucb(train_x, train_y, num_train, seed, n_arms, max_steps=max_steps)
    neurallinucb.run()
    log_regret_info(neurallinucb, 'train')

    # testing

    # 18%
    test_x, test_y = prep_shoe_data(test_data)
    dataset = ShoeDataset(x=test_x, y=test_y)
    bandit = ContextualBandit(T=num_test, n_arms=n_arms, seed=seed, dataset=dataset)

    # 54%
    # train_x, train_y = prep_shoe_data(train_data)
    # dataset = ShoeDataset(x=train_x, y=train_y)
    # bandit = ContextualBandit(T=num_train, n_arms=n_arms, seed=seed, dataset=dataset)

    neurallinucb.update_bandit(bandit)
    neurallinucb.run_inference()
    log_regret_info(neurallinucb, 'test')
    print(f'{len(neurallinucb.regrets)} test examples')
    print("Accuracy: ", 1-sum(neurallinucb.regrets)/len(neurallinucb.regrets))


if __name__=='__main__':
    main()
    