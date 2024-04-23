import numpy as np
import torch

from .model import MLP
from .ucb import UCB

from torch.utils.data import Subset
from transformers import Trainer, TrainingArguments


class NeuralLinUCB(UCB):
    """
    Neural LinUCB.
    """
    def __init__(self,
                 bandit,
                 reg_factor=1.0,
                 delta=0.01,
                 confidence_scaling_factor=-1.0,
                 training_window=200,
                 learning_rate=1e-3,
                 train_every=1,
                 throttle=1,
                 exploration_params=None,
                 num_train_epochs=3,
                 max_steps=0 # if not specified, go with epochs (which is set to 3)
                 ):
        # number of rewards in the training buffer
        self.training_window = training_window
        self.exploration_params = exploration_params

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # input dimension
        input_dim = 7
        # hidden dimension d
        hidden_dim = 32
        # output dimension
        output_dim = 6

        # neural network
        self.model = MLP(input_dim, hidden_dim, output_dim).to(self.device)
        # exclude the last layer
        params_to_optimize = [param for name, param in self.model.named_parameters() if 'fc1' in name or 'fc2' in name]
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        training_args = TrainingArguments(
            output_dir='./trained_models',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            learning_rate=learning_rate,
            # warmup_steps=500,
            # weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=100,
            save_total_limit=5,
            load_best_model_at_end=True,
            max_steps=max_steps,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            optimizers=(self.optimizer, self.lr_scheduler),
        )

        super().__init__(bandit,
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor,
                         delta=delta,
                         throttle=throttle,
                         train_every=train_every,
                         )

    @property
    def approximator_dim(self):
        """Dimension of the deberta model output.
        """
        return self.model.hidden_dim

    def reset(self):
        """Reset the internal estimates.
        """
        self.reset_upper_confidence_bounds()
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_b()
        self.iteration = 0

    def train(self):
        """Train neural network.
        """
        iterations_so_far = range(np.max([0, self.iteration-self.training_window]), self.iteration+1)
        actions_so_far = self.actions[np.max([0, self.iteration-self.training_window]):self.iteration+1]

        indices = []
        for i in range(len(iterations_so_far)):
            indices.append((iterations_so_far[i], actions_so_far[i]))

        # training set = (context, reward) for the chosen actions
        train_dataset = Subset(self.bandit.dataset, indices)

        # train model
        self.trainer.train_dataset=train_dataset
        self.trainer.eval_dataset=train_dataset # evaluate on the training set during training
        self.trainer.train()

    def predict(self):
        """Predict reward.
        """
        self.model.eval()

        # rejection reward estimation
        output = self.model(self.bandit.features[self.iteration].to(self.device),
                            torch.tensor(self.bandit.dataset.get_label(self.iteration)).to(self.device)
                            )
        pred = output['pred'].squeeze().detach().cpu().numpy()
        self.mu_hat[self.iteration] = pred

    def update_bandit(self, bandit):
        """Update bandit.
        """
        self.bandit = bandit
        self.T = self.bandit.T
        self.iteration = 0