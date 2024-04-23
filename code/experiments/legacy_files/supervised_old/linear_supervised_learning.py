# only finetune the final project layer of deberta classifier on squad
import torch
import random
import sys
import numpy as np
from transformers import TrainingArguments, Trainer

sys.path.append('..')
from utils import prep_shoe_data
from data_loader import get_data
from lin_ucb.model.linear_model import LinearModel
from lin_ucb.dataset.supervised_dataset import ShoeSupervisedDataset


def compute_accuracy(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels = np.argmax(pred.label_ids, axis=1)
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1

    acc = correct / len(predictions)
    print('acc: ', acc)

    return {
        'accuracy': acc
    }


def main():
    max_steps=1000
    per_device_batch_size=4
    learning_rate=5e-2
    task = 'shoe'
    num_train = 1000
    num_test = 100
    seed=42
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = 7
    output_dim = 6
    # neural network
    model = LinearModel(
        input_dim, output_dim
    ).to(device)

    # update all layers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=1.0) # i.e. no scheduler

    training_args = TrainingArguments(
        output_dir='./supervised_linear_model',
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        logging_dir='./supervised_linear_model_logs',
        logging_steps=100,
        evaluation_strategy='steps',
        eval_steps=100,
        save_total_limit=20, # basically no limit
        load_best_model_at_end=True,
        do_train=True,
        do_eval=True,
        max_steps=max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=(
            optimizer, 
            lr_scheduler
            ),
        compute_metrics=compute_accuracy,
    )

    # get train and dev dataset
    train_data, test_data = get_data(task, num_train, num_test)
    train_x, train_y = prep_shoe_data(train_data)
    test_x, test_y = prep_shoe_data(test_data)

    train_dataset = ShoeSupervisedDataset(x=train_x, y=train_y)
    dev_dataset = ShoeSupervisedDataset(x=test_x, y=test_y)

    # train model
    trainer.train_dataset=train_dataset
    trainer.eval_dataset=dev_dataset
    print('evaluate before training')
    trainer.evaluate()

    print('start training')
    trainer.train()

    # evaluate on training set
    print('start evaluation')
    trainer.evaluate()


if __name__=='__main__':
    main()
