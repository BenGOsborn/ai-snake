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

    # Create N generations
    for i in range(utils.GENERATIONS):
        trainer.evaluate_population()
        trainer.create_next_generation()

        if i % 5 == 0:
            print(f"i == {i} - highest fitness == {trainer.highest_fitness}")


if __name__ == "__main__":
    main()
