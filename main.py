import curses
import torch
from time import sleep
import os

from snake.snake import Snake
from display.display import Display
from model.model import Model
from model.utils import choose_key
import utils


USER_HEIGHT = 30
USER_WIDTH = 50

AI_HEIGHT = 8
AI_WIDTH = 8

FRAME_RATE = 15

GENERATION_SIZE = 250
MUTATION_CHANCE = 0.15
TRAINING_TIME_LIMIT = 1000
GENERATIONS = 100

MODEL_PATH = os.path.join(os.getcwd(), "bin", "model.pth")


def main(stdscr):
    snake = Snake(utils.AI_HEIGHT, utils.AI_WIDTH)
    display = Display(snake, stdscr)

    model = Model(utils.AI_HEIGHT * utils.AI_WIDTH)
    model.load_state_dict(torch.load(utils.MODEL_PATH))
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
