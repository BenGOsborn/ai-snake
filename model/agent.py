from model.model import Model
import model.utils as utils


class Agent:
    def __init__(self, snake, evaluations, time_limit):
        self.evaluations = evaluations
        self.time_limit = time_limit

        self.snake = snake
        self.model = Model()

        self.fitness = None

        self.model.eval()

    # Evaluate the current agent a given amount of times
    def evaluate(self):
        record = 0
        deaths = 0
        eating_times = []
        penalties = 0

        for _ in range(self.evaluations):
            time = 0

            temp_eating_times = [0]  # The times the snake ate
            prev_size = len(self.snake.snake)

            # Run the game loop
            while not self.snake.game_over() and time < self.time_limit:
                key = utils.choose_key(
                    self.snake.get_game_state(),
                    self.model
                )

                self.snake.update_state(key)

                # Record the eating times of the snake
                current_size = len(self.snake.snake)
                if current_size > prev_size:
                    temp_eating_times.append(time)
                    prev_size = current_size

                time += 1

            # Calculate score of agent
            food_consumed = len(self.snake.snake) - 1
            if food_consumed > record:
                record = food_consumed
            deaths += 1 if self.snake.game_over() else 0
            eating_times.append(temp_eating_times)
            penalties += 0 if time < self.time_limit else 1

            self.snake.reset()

        # Calculate and update the agents fitness
        self.fitness = sum(fitness) / len(fitness)
