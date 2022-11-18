import random


class Snake:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.terminated = False

        self.snake = [
            [
                random.randint(0, self.height - 1),
                random.randint(0, self.width - 1)
            ]
        ]

        self.dir = [0, 1]  # Current direction of travel

        self.food = self.select_food()

    # Get the dimensions of the game
    def get_dims(self):
        return self.height, self.width

    # Select a random location for food
    def select_food(self):
        while True:
            food = [
                random.randint(0, self.height - 1),
                random.randint(0, self.width - 1)
            ]
            if food[0] != self.snake[0][0] or food[1] != self.snake[0][1]:
                return food

    # Check if the game is over
    def game_over(self):
        return self.terminated

    # Update the state of the game
    def update_state(self, key=None):
        # Check if the game has finished
        if self.game_over():
            raise Exception("Game has finished")

        # Update position of snake
        self.snake.insert(
            0, (self.snake[0][0] + self.dir[0], self.snake[0][1] + self.dir[1])
        )
        self.snake.pop(-1)

        # Check for collision / out of bounds
        if self.snake[0] in self.snake[1:] or self.snake[0][0] in [-1, self.height] or self.snake[0][1] in [-1, self.width]:
            self.terminated = True
            return

        # Check for food and grow the snake else update snake position
        if self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]:
            self.snake.insert(0, self.food)
            self.food = self.select_food()

        # Update snake position
        if key is not None:
            if key == 0:
                self.dir = [-1, 0]  # Up
            elif key == 1:
                self.dir = [1, 0]  # Down
            elif key == 2:
                self.dir = [0, -1]  # Left
            elif key == 3:
                self.dir = [0, 1]  # Right
