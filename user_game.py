import curses
from time import sleep

from snake.snake import Snake
from display.display import Display


HEIGHT = 20
WIDTH = 20
FRAME_RATE = 10


def main(stdscr):
    snake = Snake(HEIGHT, WIDTH)
    display = Display(snake, stdscr)

    # Game loop
    while True:
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
        if snake.game_over():
            break

        display.display()

        sleep(1 / FRAME_RATE)

    curses.endwin()


if __name__ == "__main__":
    curses.wrapper(main)
