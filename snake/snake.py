import random


class Snake:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.random = random.Random(1234)

        self.terminated = False

        self.snake = None
        self.food = None

        # Initialize the game
        self.reset()

    # Reset the current game state
    def reset(self):
        self.snake = [
            (
                self.random.randint(0, self.height - 1),
                self.random.randint(0, self.width - 1)
            )
        ]

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
    def update_state(self, key):
        # Check if the game has finished
        if self.game_over():
            raise Exception("Game has finished")

        # Update snake position
        if key == 0:
            mvmnt = [-1, 0]  # Up
        elif key == 1:
            mvmnt = [1, 0]  # Down
        elif key == 2:
            mvmnt = [0, -1]  # Left
        elif key == 3:
            mvmnt = [0, 1]  # Right

        pos = (self.snake[0][0] + mvmnt[0], self.snake[0][1] + mvmnt[1])

        # Update position of snake
        if not self.is_valid_position(pos[0], pos[1]):
            self.terminated = True
            print("TERMINATED")
            return
        else:
            self.snake.insert(0, pos)

        # Check if snake encountered food
        if self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]:
            self.food = self.select_food()
        else:
            self.snake.pop(-1)
