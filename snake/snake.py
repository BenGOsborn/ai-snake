import random

import snake.utils as utils


class Snake:
    def __init__(self, height, width, food_amount, seed=None):
        self.height = height
        self.width = width
        self.food_amount = food_amount
        self.seed = seed

        self.random = None
        self.snake = None
        self.food = []

        # Initialize the game
        self.reset()

    # Reset the game
    def reset(self):
        self.random = random.Random(self.seed)
        self.choose_snake_position()
        self.choose_food_position()

    # Choose a snake position
    def choose_snake_position(self):
        self.snake = [self.select_position()]

    # Choose a food position
    def choose_food_position(self):
        for _ in range(self.food_amount - len(self.food)):
            self.food.append(self.select_position())

    # Eat food at a given position
    def eat_food(self, y, x):
        pos = -1

        for i, food in self.food:
            if (y, x) == food:
                pos = i
                break

        self.food.pop(pos)
        self.choose_food_position()

    # Select a random location for food
    def select_position(self):
        return self.random.randint(0, self.height - 1), self.random.randint(0, self.width - 1)

    # Get the game state
    def get_state(self):
        state = []

        standardized_distance = (self.height ** 2 + self.width ** 2) ** (1/2)

        # Get the distance to food and accessibility of each direction
        for y, x in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            y += self.snake[0][0]
            x += self.snake[0][1]

            state.append(self.position_value(y, x))

            print(self.position_value(y, x), (y, x))

            food_distance = max(
                [
                    (((y - food_y) ** 2 + (x - food_x) ** 2) ** (1/2)) / standardized_distance for food_x, food_y in self.food
                ]
            )
            state.append(food_distance)

        return state

    # Check if a position is accessible
    def position_value(self, y, x):
        if (y, x) in self.snake or y in [-1, self.height] or x in [-1, self.width]:
            return -1

        if (y, x) in self.food:
            return 1

        return 0

    # Update the state of the game
    def update_state(self, key):
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
        pos_value = self.position_value(*pos)

        # Update position of snake
        if pos_value == -1:
            self.choose_snake_position()

            return utils.TERMINATED, -20
        else:
            self.snake.insert(0, pos)

        # Check if snake encountered food
        if pos_value == 1:
            self.eat_food(*pos)

            return utils.ATE, 10
        else:
            self.snake.pop(-1)

            return utils.NULL, -1
