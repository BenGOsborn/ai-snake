import curses
import torch
from time import sleep
import sys

from snake.snake import Snake
from display.display import Display
from model.ga.model import GAModel
from model.utils import choose_key
import utils


def run_ga(stdscr):
    snake = Snake(
        utils.HEIGHT,
        utils.WIDTH,
        utils.FOOD_AMOUNT,
        seed=utils.SNAKE_SEED
    )

    display = Display(snake, stdscr)

    model = GAModel()
    model.load_state_dict(torch.load(utils.MODEL_PATH_GA))
    model.eval()

    # Game loop
    while True:
        key, _ = choose_key(snake.get_state(), model)

        snake.update_state(key)

        display.display()

        sleep(1 / utils.FRAME_RATE)


def run_ga_noscr():
    snake = Snake(
        utils.HEIGHT,
        utils.WIDTH,
        utils.FOOD_AMOUNT,
        seed=utils.SNAKE_SEED,
    )

    model = GAModel()
    model.load_state_dict(torch.load(utils.MODEL_PATH_GA))
    model.eval()

    # Game loop
    while True:
        state = snake.get_state()
        key, _ = choose_key(state, model)

        print(
            f"Snake size {len(snake.snake)} - Snake head {snake.snake[0]} - Food {snake.food}\nState - {state}"
        )

        snake.update_state(key)

        sleep(1 / utils.FRAME_RATE)


if __name__ == "__main__":
    if "ga" in sys.argv:
        if "d" in sys.argv:
            curses.wrapper(run_ga)
        else:
            run_ga_noscr()
    elif "dqn" in sys.argv:
        pass
