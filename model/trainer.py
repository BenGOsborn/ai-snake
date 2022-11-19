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
        # **** We will create a mask where if it is true we will select state 1 and if it is false we will select state 2
        # **** Then we will create a second mask which will be for a completely random weight (mutation)

        state1 = agent1.model.state_dict()
        state2 = agent2.model.state_dict()

        for key in state1:
            print(key)

    # Create the next generation
    def create_next_generation(self):
        self.breed(self.generation[0], self.generation[1])
