import curses
from time import sleep

from snake.snake import Snake


FRAME_RATE = 30

HEIGHT = 25
WIDTH = 25


def main(stdscr):
    stdscr.nodelay(1)

    snake = Snake(HEIGHT, WIDTH)

    # Game loop
    while not snake.game_over():
        # Get input
        try:
            key = stdscr.getkey()
        except curses.error:
            key = None

        # Update game state
        input_key = None
        if key == "KEY_UP":
            input_key = 0
        elif key == "KEY_DOWN":
            input_key = 1
        elif key == "KEY_LEFT":
            input_key = 2
        elif key == "KEY_RIGHT":
            input_key = 3

        snake.update_state(key=input_key)

        # Draw to screen
        stdscr.clear()

        stdscr.addch(snake.food[0], snake.food[1], curses.ACS_PI)

        for body in snake.snake:
            stdscr.addch(body[0], body[1], curses.ACS_BLOCK)

        stdscr.refresh()

        # Wait for next frame
        sleep(1 / FRAME_RATE)

    curses.endwin()


if __name__ == "__main__":
    curses.wrapper(main)
