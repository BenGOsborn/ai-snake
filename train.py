from snake.snake import Snake
from model.ga.trainer import GATrainer
from model.dqn.trainer import DQNTrainer
import utils


def train_ga():
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


def train_dqn():
    # Initialize
    snake = Snake(
        utils.HEIGHT,
        utils.WIDTH,
        utils.FOOD_AMOUNT,
        seed=utils.SNAKE_SEED
    )

    trainer = DQNTrainer(snake)

    # Train for N generations
    for i in range(utils.GENERATIONS):
        trainer.train_step()

        if i % 20 == 0:
            trainer.save_model(utils.MODEL_PATH_DQN)


if __name__ == "__main__":
    train_ga()
