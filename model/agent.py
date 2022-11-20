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

        scores = []
        current_score = 0
        start_time = 0

        # Run the game loop
        while time < time_limit:
            # Choose a key and update the state
            key = model_utils.choose_key(self.snake.get_state(), self.model)
            event, _ = self.snake.update_state(key)

            # Update time
            time += 1

            # Update score
            if event == snake_utils.ATE:
                current_score += 1
            elif event == snake_utils.TERMINATED:
                scores.append(current_score / (time - start_time))
                current_score = 0
                start_time = time

        # Update agents fitness
        scores = scores + [current_score / (time + 1 - start_time)]

        self.fitness = sum(scores) / len(scores)
