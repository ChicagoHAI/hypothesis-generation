import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        y_hat = self.linear(x)

        loss = self.loss(y_hat, labels)
        
        # print('loss: ', loss)

        return {
            'pred': y_hat,
            'loss': loss
        }
