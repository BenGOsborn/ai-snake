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

    trainer.evaluate_population()
    trainer.create_next_generation()


if __name__ == "__main__":
    main()
