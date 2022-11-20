import curses
import torch
from time import sleep

from snake.snake import Snake
from display.display import Display
from model.model import Model
from model.utils import choose_key
import utils


def main(stdscr):
    snake = Snake(utils.HEIGHT, utils.WIDTH)
    display = Display(snake, stdscr)

    model = Model()
    model.load_state_dict(torch.load(utils.MODEL_PATH_GA))
    model.eval()

    # Game loop
    while True:
        input_key = choose_key(snake.get_state(), model)

        snake.update_state(input_key)

        display.display()

        sleep(1 / utils.FRAME_RATE)


def main_noscrn():
    snake = Snake(utils.HEIGHT, utils.WIDTH)

    model = Model()
    model.load_state_dict(torch.load(utils.MODEL_PATH))
    model.eval()

    # Game loop
    while True:
        input_key = choose_key(snake.get_state(), model)

        snake.update_state(input_key)

        sleep(1 / utils.FRAME_RATE)


if __name__ == "__main__":
    curses.wrapper(main)
    # main_noscrn()
