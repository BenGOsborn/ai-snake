import os


# General params
FRAME_RATE = 10

# Snake params
SNAKE_SEED = None
HEIGHT = 10
WIDTH = 10
FOOD_AMOUNT = 12

# Genetic algorithm params
GA_GENERATIONS = 200
GENERATION_SIZE = 100
TOP_K = 25
MUTATION_CHANCE = 0.15
MUTATE_POP_CHANCE = 0.20

# DQN params
DQN_EPOCHS = 100000

# Model paths
MODEL_PATH_GA = os.path.join(os.getcwd(), "bin", "model_ga.pth")
MODEL_PATH_DQN = os.path.join(os.getcwd(), "bin", "model_dqn.pth")
