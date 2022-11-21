import torch.nn as nn


class GAModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        return self.network(inputs)
