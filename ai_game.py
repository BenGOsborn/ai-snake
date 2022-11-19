import curses
from time import sleep

from snake.snake import Snake
from display.display import Display
from model.model import Model
from model.utils import choose_key
import utils


def main(stdscr):
    snake = Snake(utils.AI_HEIGHT, utils.AI_WIDTH)
    display = Display(snake, stdscr)

    model = Model(utils.AI_HEIGHT * utils.AI_WIDTH)
    model.eval()

    # Game loop
    while True:
        input_key = choose_key(snake.get_game_state(), model)

        snake.update_state(key=input_key)
        if snake.game_over():
            break

        display.display()

        sleep(1 / utils.FRAME_RATE)

    curses.endwin()


if __name__ == "__main__":
    curses.wrapper(main)
