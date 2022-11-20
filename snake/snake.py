import random


class Snake:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.random = random.Random()

        self.snake = None
        self.food = self.food = self.select_position()

        # Initialize the snake
        self.reset()

    # Reset the snake
    def reset(self):
        self.snake = [self.select_position()]

    # Select a random location for food
    def select_position(self):
        return self.random.randint(0, self.height - 1), self.random.randint(0, self.width - 1)

    # Get the game state
    def get_state(self):
        state = []

        standardized_distance = (self.height ** 2 + self.width ** 2) ** (1/2)
        food_y, food_x = self.food

        # Get the distance to food and accessibility of each direction
        for y, x in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            y += self.snake[0][0]
            x += self.snake[0][1]

            blocked_pos = 0 if self.is_valid_position(y, x) else 1
            state.append(blocked_pos)

            food_distance = (
                ((y - food_y) ** 2 + (x - food_x) ** 2) ** (1/2)
            ) / standardized_distance
            state.append(food_distance)

        return state

    # Check if a position is accessible
    def is_valid_position(self, y, x):
        return not ((y, x) in self.snake or y in [-1, self.height] or x in [-1, self.width])

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

        # Update position of snake
        if not self.is_valid_position(pos[0], pos[1]):
            self.reset()

            return -1, -20
        else:
            self.snake.insert(0, pos)

        # Check if snake encountered food
        if self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]:
            self.food = self.select_position()

            return 1, 10
        else:
            self.snake.pop(-1)

            return 0, -1
