import random

import snake.utils as utils


class Snake:
    def __init__(self, height, width, food_amount, seed=None, stuck_limit=100):
        self.height = height
        self.width = width
        self.food_amount = food_amount
        self.seed = seed
        self.stuck_limit = stuck_limit

        self.random = None
        self.snake = None
        self.food = []

        self.counter = 0
        self.last_eaten = 0

        self.dir = (1, 0)

        # Initialize the game
        self.reset()

    # Reset the game
    def reset(self):
        self.counter = 0
        self.last_eaten = 0

        self.random = random.Random(self.seed)
        self.choose_snake_position()
        self.choose_food_position()

    # Choose a snake position
    def choose_snake_position(self):
        self.snake = [self.select_position()]

    # Choose a food position
    def choose_food_position(self):
        for _ in range(self.food_amount - len(self.food)):
            position = self.select_position()
            self.food.append(position)
            print(f"New food position: {position}")

    # Eat food at a given position
    def eat_food(self, y, x):
        pos = -1

        for i, food in enumerate(self.food):
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
        for y, x in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            y += self.snake[0][0]
            x += self.snake[0][1]

            state.append(self.pos_value(y, x))

            food_distance = min(
                [
                    (((y - food_y) ** 2 + (x - food_x) ** 2) ** (1/2)) / standardized_distance for food_x, food_y in self.food
                ]
            )
            state.append(food_distance)

        # Add the context of the previous snake position
        for pos in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            state.append(1 if pos == self.dir else 0)

        return state

    # Check if a position is accessible
    def pos_value(self, y, x):
        if (y, x) in self.snake or y in [-1, self.height] or x in [-1, self.width]:
            return -1

        if (y, x) in self.food:
            return 1

        return 0

    # Update the state of the game
    def update_state(self, key):
        # Reset the snake if it is stuck
        if self.counter - self.last_eaten == self.stuck_limit:
            self.choose_snake_position()

            self.last_eaten = self.counter

            return utils.STUCK

        # Update snake position
        if key == 0:
            self.dir = (-1, 0)  # Up
        elif key == 1:
            self.dir = (1, 0)  # Down
        elif key == 2:
            self.dir = (0, -1)  # Left
        elif key == 3:
            self.dir = (0, 1)  # Right

        pos = (self.snake[0][0] + self.dir[0], self.snake[0][1] + self.dir[1])
        pos_value = self.pos_value(*pos)

        # Update position of snake
        if pos_value == -1:
            self.choose_snake_position()

            self.last_eaten = self.counter

            return utils.TERMINATED
        else:
            self.snake.insert(0, pos)

        # Check if snake encountered food
        if pos_value == 1:
            self.eat_food(*pos)

            self.last_eaten = self.counter

            return utils.ATE
        else:
            self.snake.pop(-1)

            self.counter += 1

            return utils.NULL
