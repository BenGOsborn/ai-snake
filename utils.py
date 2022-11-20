import os


HEIGHT = 8
WIDTH = 8
FRAME_RATE = 15

GENERATION_SIZE = 200
TOP_AGENTS = 30
MUTATION_CHANCE = 0.1
EVALUATIONS = 10
TRAINING_TIME_LIMIT = 1000
STUCK_LIMIT = 75
GENERATIONS = 10000

MODEL_PATH_NEAT = os.path.join(os.getcwd(), "bin", "model_neat.pth")
