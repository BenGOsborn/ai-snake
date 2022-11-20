import model.utils as model_utils
import snake.utils as snake_utils


class Agent:
    def __init__(self, snake, model):
        self.snake = snake
        self.model = model

        self.fitness = None

    # Evaluate the agent and get its fitness
    def evaluate(self, time_limit=1000):
        # Reset the snake
        self.snake.reset()

        time = 0

        total_score = 0

        # Run the game loop
        while time < time_limit:
            # Choose a key and update the state
            key = model_utils.choose_key(self.snake.get_state(), self.model)
            event, _ = self.snake.update_state(key)

            if event == snake_utils.ATE:
                total_score += 1

            # Update time
            time += 1

        # Calculate and update the agents fitness
        self.fitness = total_score / time
