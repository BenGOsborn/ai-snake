from snake.snake import Snake
from model.brain import Brain


class Agent:
    def __init__(self, height, width):
        self.snake = Snake(height, width)
        self.brain = Brain(self.snake)

        self.fitness = None

    # Evaluate the current agent
    def evaluate(self):
        time = 0

        while not self.snake.game_over():
            time += 1

        self.fitness = len(self.snake.snake) / time
