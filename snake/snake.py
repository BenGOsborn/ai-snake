class Snake:
    def __init__(self, height, width, seed):
        self.height = height
        self.width = width

        self.seed = seed

        self.terminated = False

        self.snake = [
            [
                self.randint(self.height),
                self.randint(self.width)
            ]
        ]

        self.dir = [0, 1]  # Current direction of travel

        self.food = self.select_food()

    # Get a pseudorandom integer
    def randint(self, size):
        hash_value = hash(self.seed)
        self.seed += 1

        return hash_value % size

    # Get the game board
    def get_game_state(self):
        board = [[0 for _ in range(self.width)] for _ in range(self.height)]

        for y, x in self.snake[1:]:
            board[y][x] = 0.5

        board[self.snake[0][0]][self.snake[0][1]] = 1
        board[self.food[0]][self.food[1]] = -1

        return board

    # Select a random location for food
    def select_food(self):
        while True:
            food = [
                self.randint(self.height),
                self.randint(self.width)
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
