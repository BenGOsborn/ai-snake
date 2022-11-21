import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, inputs):
        return self.network(inputs)
