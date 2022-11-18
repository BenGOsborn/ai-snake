from model.trainer import Trainer
import utils


def main():
    trainer = Trainer(utils.AI_HEIGHT, utils.AI_WIDTH, utils.GENERATION_SIZE)

    trainer.evaluate_population()


if __name__ == "__main__":
    main()
