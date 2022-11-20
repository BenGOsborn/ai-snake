from statistics import mean

import model.utils as utils


class Agent:
    def __init__(self, snake, model):
        self.snake = snake
        self.model = model

        self.fitness = None

    # Evaluate the current agent
    def evaluate(self, time_limit=1000, stuck_limit=100):
        time = 0

        eating_times = [0]  # The times the snake ate
        prev_size = len(self.snake.snake)
        penalty = 0

        # Run the game loop
        while not self.snake.game_over() and time < time_limit:
            # Check if the snake is stuck
            if time - eating_times[-1] == stuck_limit:
                penalty = 1
                break

            # Choose a key and update the state
            key = utils.choose_key(
                self.snake.get_state(),
                self.model
            )
            self.snake.update_state(key)

            # Record the eating times of the snake
            current_size = len(self.snake.snake)
            if current_size > prev_size:
                eating_times.append(time)
                prev_size = current_size

            # Update the time
            time += 1

        # Calculate score of agent
        fitness = len(self.snake.snake) - 1
        deaths = 1 if self.snake.game_over() else 0
        steps = [eating_times[i] - eating_times[i - 1]
                 for i in range(1, len(eating_times))]

        # Reset the snake
        self.snake.reset()

        # Calculate and update the agents fitness
        self.fitness = fitness * 5
        self.fitness = self.fitness - deaths * 0.5 - penalty * 1
        self.fitness = self.fitness - \
            mean(steps) * 0.1 if len(steps) > 0 else self.fitness
