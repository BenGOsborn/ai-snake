import random


class Snake:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # **** Could be optimized by using a LinkedList (the head is the -1 element)
        self.snake = [[self.height // 2, self.width // 2]]
        self.food = self.select_food()

        self.dir = [0, 1]

        self.terminated = False

    # Get the dimensions of the game
    def get_dims(self):
        return self.height, self.width

    # Select a random location for food
    # ***** This can be better optimized using a reverse mapping
    def select_food(self):
        food = None

        while True:
            food = [
                random.randint(0, self.height - 1),
                random.randint(0, self.width - 1)
            ]
            if food not in self.snake:
                return food

    # Check if the game is over
    def game_over(self):
        return self.terminated

    # Update the state of the game
    def update_state(self, key=None):
        if self.game_over():
            raise Exception("Game has already finished")

        # Check for collision / out of bounds
        if self.snake[0] in self.snake[1:] or self.snake[0][0] in [-1, self.height] or self.snake[0][1] in [-1, self.width]:
            self.terminated = True
            return

        # Check for food and grow the snake
        if self.snake[0] == self.food:
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

        self.snake.insert(
            0, (self.snake[0] + self.dir[0], self.snake[1] + self.dir[1])
        )
        self.snake.pop(-1)
