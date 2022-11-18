import random


class Snake:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.snake = [[self.width // 2, self.height // 2]]
        self.food = self.select_food()

        self.dir = [1, 0]

        self.terminated = False

    # Get the dimensions of the game
    def get_dims(self):
        return self.width, self.height

    # Select a random location for food
    def select_food(self):
        return random.randint(0, self.height - 1), random.randint(0, self.width - 1)

    # Check if the game is over
    def game_over(self):
        return self.terminated

    # Update the state of the game
    def update_state(self, key=None):
        # **** Check for food
        # **** Check for collision / out of bounds
        # **** Update position of snake
        # **** Terminate game if necessary

        pass
