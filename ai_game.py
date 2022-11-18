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
        # **** Now in here we have the model controlling the snake logic
        input_key = None

        snake.update_state(key=input_key)
        if snake.game_over():
            break

        display.display()

        sleep(1 / FRAME_RATE)

    curses.endwin()


if __name__ == "__main__":
    curses.wrapper(main)
