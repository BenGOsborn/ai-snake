import torch


# Choose a key from a single game state
def choose_key(state, model):
    inputs = torch.tensor(
        state,
        dtype=torch.float
    ).unsqueeze(0)

    with torch.no_grad():
        vals = model(inputs)

    argmax = torch.argmax(vals)

    return argmax, vals.squeeze()[argmax]
