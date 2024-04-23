import torch.nn as nn

# Define a simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        self.theta = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)

        x = self.theta(x)
        y_hat = self.sigmoid(x)

        # print('x: ', x)
        # print('y_hat: ', y_hat)
        # print('labels: ', labels)

        loss = self.loss(y_hat, labels)
        
        # print('loss: ', loss)

        return {
            'pred': y_hat,
            'loss': loss
        }
    
    def last_hidden_output(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        
        return x