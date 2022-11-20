import model.utils as utils


class Agent:
    def __init__(self, snake, model):
        self.snake = snake
        self.model = model

        self.fitness = None

    # # Evaluate the agent and get its fitness
    # def evaluate(self, time_limit=250, stuck_limit=50):
    #     # Reset the snake
    #     self.snake.reset()

    #     time = 0

    #     record = 0

    #     # Run the game loop
    #     while time < time_limit:
    #         # Choose a key and update the state
    #         key = utils.choose_key(self.snake.get_state(), self.model)
    #         self.snake.update_state(key)

    #         # Update record
    #         score = len(self.snake.snake) - 1

    #         if score > record:
    #             record = score

    #         # Update time
    #         time += 1

    #     # Calculate and update the agents fitness
    #     self.fitness = record

    def evaluate(self, time_limit=1000, stuck_limit=100):
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
            key = utils.choose_key(self.snake.get_state(), self.model)
            event, _ = self.snake.update_state(key)

            # Process eating and death events
            if event == -1:
                deaths += 1
            elif event == 1:
                eating_times.append(time)

            # Update record
            current_food = len(self.snake.snake) - 1

            if current_food > record:
                record = current_food

            # Update the time
            time += 1

            # Check if the snake is stuck
            if (time - eating_times[-1]) % stuck_limit == 0:
                penalty += 1
                self.snake.reset()

        # Calculate score of agent
        steps = [
            eating_times[i] - eating_times[i - 1] for i in range(1, len(eating_times))
        ]

        # Calculate and update the agents fitness
        self.fitness = record * 5
        self.fitness = self.fitness - deaths * 0.15 - penalty
        self.fitness = self.fitness - \
            (sum(steps) / len(steps)) * 0.1 if len(steps) > 0 else self.fitness
