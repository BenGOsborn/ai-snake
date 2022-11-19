import os

from model.trainer import Trainer
import utils


def main():
    trainer = Trainer(
        utils.AI_HEIGHT,
        utils.AI_WIDTH,
        utils.GENERATION_SIZE,
        utils.MUTATION_CHANCE,
        utils.TRAINING_TIME_LIMIT,
    )

    prev_best = -1

    # Create N generations
    for i in range(utils.GENERATIONS):
        trainer.evaluate_population()
        trainer.create_next_generation()

        if (i + 1) % 5 == 0:
            current_best = trainer.best_fitness

            print(f"i == {i} - best fitness == {current_best}")

            if current_best > prev_best:
                print(f"Saving to '{utils.MODEL_PATH}'...")

                trainer.save_best_agent(utils.MODEL_PATH)
                prev_best = current_best


if __name__ == "__main__":
    main()
