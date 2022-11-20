import model.utils as utils


class Agent:
    def __init__(self, snake, model):
        self.snake = snake
        self.model = model

        self.fitness = None

    # Evaluate the agent and get its fitness
    def evaluate(self, time_limit=250):
        # Reset the snake
        self.snake.reset()

        time = 0

        record = 0

        # Run the game loop
        while time < time_limit:
            # Choose a key and update the state
            key = utils.choose_key(self.snake.get_state(), self.model)
            self.snake.update_state(key)

            # Update record
            score = len(self.snake.snake) - 1

            if score > record:
                record = score

            # Update time
            time += 1

        # Calculate and update the agents fitness
        self.fitness = record
