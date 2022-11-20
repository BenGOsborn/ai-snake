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
    prev_global_best = -1

    for i in range(utils.GENERATIONS):
        gen_avg, gen_best = trainer.evaluate_population()
        trainer.create_next_generation()

        if i % 5 == 0:
            global_best = trainer.best_fitness

            print(
                f"NEAT - generation {i} - average {gen_avg} - best {gen_best} - global best {global_best}"
            )

            if global_best > prev_global_best:
                print(f"Saving to '{utils.MODEL_PATH_NEAT}'...")

                trainer.save_best_agent(utils.MODEL_PATH_NEAT)

                prev_global_best = global_best


if __name__ == "__main__":
    main()
