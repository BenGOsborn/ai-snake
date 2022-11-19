from snake.snake import Snake
from model.model import Model
import model.utils as utils


class Agent:
    def __init__(self, height, width, time_limit):
        self.time_limit = time_limit

        self.snake = Snake(height, width)
        self.model = Model(height * width)

        self.fitness = None

    # Evaluate the current agent
    def evaluate(self):
        time = 0

        # Complete game loop
        while not self.snake.game_over() and time < self.time_limit:
            key = utils.choose_key(
                self.snake.get_game_state(),
                self.model
            )

            self.snake.update_state(key)

            time += 1

        # Calculate score of agent
        self.fitness = len(self.snake.snake)
