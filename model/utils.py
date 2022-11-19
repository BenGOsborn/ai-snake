import torch


# Let the model choose a key based on the game state
def choose_key(snake_game_state, model):
    inputs = torch.tensor(
        snake_game_state,
        dtype=torch.float
    ).unsqueeze(0)

    with torch.no_grad():
        probs = model(inputs)

    pos = torch.argmax(probs).item()

    return pos if pos != 4 else None  # 4th index is no move
