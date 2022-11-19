from snake.snake import Snake
from model.brain import Brain


class Agent:
    def __init__(self, height, width, time_limit=1000):
        self.time_limit = time_limit

        self.snake = Snake(height, width)
        self.brain = Brain(self.snake)

        self.fitness = None

    # Evaluate the current agent
    def evaluate(self):
        time = 0

        # Complete game loop
        while not self.snake.game_over() and time < self.time_limit:
            key = self.brain.choose_key()

            self.snake.update_state(key)

            time += 1

        # Calculate score of agent
        self.fitness = len(self.snake.snake)
