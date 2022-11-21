import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        return self.network(inputs)
