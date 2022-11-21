import os


# General params
FRAME_RATE = 10

# Snake params
SNAKE_SEED = None
HEIGHT = 10
WIDTH = 10
FOOD_AMOUNT = 12

# Genetic algorithm params
GENERATION_SIZE = 100
TOP_K = 25
MUTATION_CHANCE = 0.15
MUTATE_POP_CHANCE = 0.20
GENERATIONS = 200

MODEL_PATH_GA = os.path.join(os.getcwd(), "bin", "model_ga.pth")
