from snake.snake import Snake
from model.trainer import Trainer
import utils


def main():
    # Initialize
    snake = Snake(utils.HEIGHT, utils.WIDTH)

    trainer = Trainer(
        snake,
        utils.GENERATION_SIZE,
        utils.TOP_AGENTS,
        utils.MUTATION_CHANCE,
        utils.EVALUATIONS,
        utils.TRAINING_TIME_LIMIT,
        utils.STUCK_LIMIT,
    )

    # Train for N generations
    for i in range(utils.GENERATIONS):
        trainer.evaluate_population()
        trainer.create_next_generation()

        if i % 20 == 0:
            print(f"Generation {i} - saving to '{utils.MODEL_PATH_NEAT}'...")

            trainer.save_best_agent(utils.MODEL_PATH_NEAT)


if __name__ == "__main__":
    main()
