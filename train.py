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

    prev_best = -1

    # Create N generations
    for i in range(utils.GENERATIONS):
        trainer.evaluate_population()
        trainer.create_next_generation()

        if i % 5 == 0:
            current_best = trainer.best_fitness

            print(f"NEAT - generation {i} - best fitness {current_best}")

            if current_best > prev_best:
                print(f"Saving to '{utils.MODEL_PATH_NEAT}'...")

                trainer.save_best_agent(utils.MODEL_PATH_NEAT)

                prev_best = current_best


if __name__ == "__main__":
    main()
