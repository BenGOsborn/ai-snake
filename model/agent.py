from statistics import mean

import model.utils as utils


class Agent:
    def __init__(self, snake, model):
        self.snake = snake
        self.model = model

        self.fitness = None

    # Evaluate the current agent
    def evaluate(self, time_limit=1000, stuck_limit=100):
        # Reset the snake
        self.snake.reset()

        time = 0

        record = 0
        eating_times = [0]  # The times the snake ate
        penalty = 0

        # Run the game loop
        while time < time_limit:
            # Check if the snake is stuck
            if (time - eating_times[-1]) % stuck_limit == 0:
                penalty += 1
                self.snake.reset()
                continue

            # Choose a key and update the state
            key = utils.choose_key(
                self.snake.get_state(),
                self.model
            )
            self.snake.update_state(key)

            # Record the eating times of the snake
            current_food = len(self.snake.snake) - 1

            if current_food > record:
                record = current_food

            # Update the time
            time += 1

        # Calculate score of agent
        record = len(self.snake.snake) - 1
        deaths = 1 if self.snake.game_over() else 0
        steps = [
            eating_times[i] - eating_times[i - 1] for i in range(1, len(eating_times))
        ]

        # Calculate and update the agents fitness
        self.fitness = record * 5
        self.fitness = self.fitness - deaths * 0.5 - penalty * 1
        self.fitness = self.fitness - \
            mean(steps) * 0.1 if len(steps) > 0 else self.fitness
