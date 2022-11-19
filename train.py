import os

from model.trainer import Trainer
import utils


def main():
    trainer = Trainer(
        utils.HEIGHT,
        utils.WIDTH,
        utils.GENERATION_SIZE,
        utils.MUTATION_CHANCE,
        utils.EVALUATIONS,
        utils.TRAINING_TIME_LIMIT,
    )

    # Create N generations
    for i in range(utils.GENERATIONS):
        trainer.evaluate_population()
        trainer.create_next_generation()

        if i % 5 == 0:
            current_best = trainer.best_fitness

            print(f"i == {i} - best fitness == {current_best}")

    print(f"Finished - saving to '{utils.MODEL_PATH}'...")

    trainer.save_best_agent(utils.MODEL_PATH)


if __name__ == "__main__":
    main()
