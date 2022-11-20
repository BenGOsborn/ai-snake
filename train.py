from snake.snake import Snake
from model.trainer import Trainer
import utils


def main():
    # Initialize
    snake = Snake(utils.HEIGHT, utils.WIDTH)

    trainer = Trainer(
        snake,
        utils.GENERATION_SIZE,
        utils.MUTATION_CHANCE,
        utils.MUTATE_POP_CHANCE,
    )

    # Train for N generations
    for i in range(utils.GENERATIONS):
        trainer.evaluate_population()
        trainer.create_next_generation()

        if i % 20 == 0:
            print(f"Generation {i} - saving to '{utils.MODEL_PATH_GA}'...")

            trainer.save_best_agent(utils.MODEL_PATH_GA)


if __name__ == "__main__":
    main()
