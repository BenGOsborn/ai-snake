from snake.snake import Snake
from model.model import Model
import model.utils as utils


class Agent:
    def __init__(self, height, width, evaluations, time_limit):
        self.evaluations = evaluations
        self.time_limit = time_limit

        self.snake = Snake(height, width)
        self.model = Model()

        self.fitness = None

        self.model.eval()

    # Evaluate the current agent
    def evaluate(self):
        # Evaluate performance over n evaluations
        fitness = []

        for _ in self.evaluations:
            time = 0

            while not self.snake.game_over() and time < self.time_limit:
                key = utils.choose_key(
                    self.snake.get_game_state(),
                    self.model
                )

                self.snake.update_state(key)

                time += 1

            # Calculate score of agent
            fitness_score = len(self.snake.snake)
            fitness.append(fitness_score)

            self.snake.reset()

        self.fitness = sum(fitness) / len(fitness)
