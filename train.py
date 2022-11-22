from snake.snake import Snake
from model.ga.trainer import GATrainer
import utils


def main():
    # Initialize
    snake = Snake(
        utils.HEIGHT,
        utils.WIDTH,
        utils.FOOD_AMOUNT,
        seed=utils.SNAKE_SEED
    )

    trainer = GATrainer(
        snake,
        utils.GENERATION_SIZE,
        utils.TOP_K,
        utils.MUTATION_CHANCE,
        utils.MUTATE_POP_CHANCE,
    )

    # Train for N generations
    for i in range(utils.GENERATIONS):
        trainer.evaluate_population()
        trainer.create_next_generation()

        if i % 20 == 0:
            trainer.save_best_agent(utils.MODEL_PATH_GA)


if __name__ == "__main__":
    main()
