import torch


# Let the model choose a key based on the game state
def choose_key(state, model):
    inputs = torch.tensor(
        state,
        dtype=torch.float
    ).unsqueeze(0)

    with torch.no_grad():
        probs = model(inputs)

    return torch.argmax(probs).item()
