import curses


class Display:
    def __init__(self, snake, stdscr):
        self.snake = snake
        self.stdscr = stdscr

        self.stdscr.nodelay(1)

    # Draw board to screen
    def display(self):
        self.stdscr.clear()

        # Draw snake
        self.stdscr.addch(
            self.snake.snake[0][0], self.snake.snake[0][1], curses.ACS_DIAMOND)

        for body in self.snake.snake[1:]:
            self.stdscr.addch(body[0], body[1], curses.ACS_BLOCK)

        # Draw food
        for food in self.snake.food:
            self.stdscr.addch(food[0], food[1], curses.ACS_PI)

        # Draw border Y
        for i in range(self.snake.height):
            self.stdscr.addch(i, self.snake.width, curses.ACS_PLUS)

        # Draw border X
        for i in range(self.snake.width):
            self.stdscr.addch(self.snake.height, i, curses.ACS_PLUS)

        self.stdscr.refresh()
