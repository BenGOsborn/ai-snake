import os


SEED = None
FRAME_RATE = 15

HEIGHT = 10
WIDTH = 10
FOOD_AMOUNT = 6
STUCK_LIMIT = 100

GENERATION_SIZE = 100
TOP_K = 25
MUTATION_CHANCE = 0.15
MUTATE_POP_CHANCE = 0.20
GENERATIONS = 10000

MODEL_PATH_GA = os.path.join(os.getcwd(), "bin", "model_ga.pth")
