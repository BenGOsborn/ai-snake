import random


class Snake:
    def __init__(self, height, width, seed):
        self.height = height
        self.width = width
        self.random = random.Random(seed)

        self.terminated = False

        # Initialize the snake
        self.snake = [
            [
                self.random.randint(0, self.height - 1),
                self.random.randint(0, self.width - 1)
            ]
        ]

        # Current direction of travel
        self.dir = [0, 1]

        # Initialize food
        self.food = self.select_food()

    # Get the game state
    def get_game_state(self):
        state = []

        standardized_distance = (self.height ** 2 + self.width ** 2) ** (1/2)
        food_y, food_x = self.food

        # Get the distance to food and accessibility of each direction
        for y, x in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            y += self.snake[0][0]
            x += self.snake[0][1]

            is_available = 1 if self.is_valid_position(y, x) else 0
            state.append(is_available)

            if is_available == 0:
                print("IT IS ZERO")

            food_distance = (
                ((y - food_y) ** 2 + (x - food_x) ** 2) ** (1/2)
            ) / standardized_distance
            state.append(food_distance)

        return state

    # Select a random location for food
    def select_food(self):
        while True:
            food = [
                self.random.randint(0, self.height - 1),
                self.random.randint(0, self.width - 1)
            ]
            if food[0] != self.snake[0][0] or food[1] != self.snake[0][1]:
                return food

    # Check if the game is over
    def game_over(self):
        return self.terminated

    # Check if a position is accessible
    def is_valid_position(self, y, x):
        return not ([y, x] in self.snake or y in [-1, self.height] or x in [-1, self.width])

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
        if not self.is_valid_position(self.snake[0][0], self.snake[0][1]):
            self.terminated = True
            return

        # Check for food and grow the snake else update snake position
        if self.snake[0] in [self.food]:
            self.snake.insert(0, self.food)
            self.food = self.select_food()

        # Update snake position
        if key == 0:
            self.dir = [-1, 0]  # Up
        elif key == 1:
            self.dir = [1, 0]  # Down
        elif key == 2:
            self.dir = [0, -1]  # Left
        elif key == 3:
            self.dir = [0, 1]  # Right
