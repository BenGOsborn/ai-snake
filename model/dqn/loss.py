import torch.nn as nn


mse_loss = nn.MSELoss()


# Loss function
def loss_fn(reward, q_max, gamma, prediction):
    target = reward + gamma * q_max

    return mse_loss(target, prediction)
