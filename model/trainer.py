import torch.nn as nn

from model.agent import Agent


class Trainer:
    def __init__(self, height, width, generation_size, mutation_chance):
        self.generation_size = generation_size
        self.mutation_chance = mutation_chance

        self.generation = [
            Agent(height, width) for _ in range(generation_size)
        ]

    # Evaluate all agents in the current population
    def evaluate_population(self):
        for elem in self.generation:
            elem.evaluate()

    # Breed two agents together
    def breed(self, agent1, agent2):
        pass

    # Create the next generation
    def create_next_generation(self):
        pass
