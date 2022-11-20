import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        return self.network(inputs)
