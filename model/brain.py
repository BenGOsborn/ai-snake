import torch
from model.model import Model


class Brain:
    def __init__(self, snake):
        self.snake = snake
        self.model = Model(snake.width * snake.height)

    # Load a model
    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))

    # Save the model
    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    # Choose a key for the snake to move
    def choose_key(self):
        inputs = torch.tensor(
            self.snake.get_game_state(),
            dtype=torch.float
        ).unsqueeze(0)

        with torch.no_grad():
            probs = self.model(inputs)

        pos = torch.argmax(probs).item()

        return pos if pos != 4 else None  # 4th index is no move
