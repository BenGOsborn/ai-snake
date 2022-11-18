import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.Softmax()
        )

    def forward(self, inputs):
        return self.network(inputs)


if __name__ == "__main__":
    model = Model(100)

    print(model.state_dict())
