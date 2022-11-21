import model.utils as model_utils
import snake.utils as snake_utils


class Agent:
    def __init__(self, snake, model):
        self.snake = snake
        self.model = model

        self.fitness = None

    # Evaluate the current agent
    def evaluate(self, time_limit=1000):
        # Reset the snake
        self.snake.reset()

        time = 0

        record = 0
        deaths = 0
        penalty = 0
        eating_times = [0]  # The times the snake ate

        # Run the game loop
        while time < time_limit:
            # Choose a key and update the state
            key = model_utils.choose_key(self.snake.get_state(), self.model)
            event = self.snake.update_state(key)

            # Process eating and death events
            if event == snake_utils.TERMINATED:
                deaths += 1
            elif event == snake_utils.ATE:
                eating_times.append(time)
            elif event == snake_utils.STUCK:
                penalty += 1

            # Update record
            record = max(record, len(self.snake.snake) - 1)

            # Update the time
            time += 1

        # Calculate score of agent
        steps = [
            eating_times[i] - eating_times[i - 1] for i in range(1, len(eating_times))
        ]
        avg_steps = sum(steps) / len(steps) if len(steps) > 0 else 0

        # Calculate and update the agents fitness
        self.fitness = record * 5
        self.fitness = self.fitness - deaths - penalty * 5 - avg_steps * 0.1
