import torch

from model.agent import Agent


class Trainer:
    def __init__(self, height, width, generation_size, mutation_chance, time_limit):
        self.generation_size = generation_size
        self.mutation_chance = mutation_chance
        self.time_limit = time_limit
        self.height = height
        self.width = width

        self.generation = [
            Agent(height, width, time_limit) for _ in range(generation_size)
        ]

    # Evaluate all agents in the current population
    def evaluate_population(self):
        for elem in self.generation:
            elem.evaluate()

    # Breed two agents together
    def breed(self, agent1, agent2):
        state1 = agent1.model.state_dict()
        state2 = agent2.model.state_dict()

        # Merge genes together randomly
        new_genes = {}

        mutation_bound = [
            -self.mutation_chance / 2,
            self.mutation_chance / 2
        ]

        # Mutate all genes
        for key in state1:
            rand_mask = 2 * torch.rand(state1[key].shape) - 1

            genes1 = (rand_mask <= mutation_bound[0]) * state1[key]
            genes2 = (rand_mask >= mutation_bound[1]) * state2[key]
            genes_mutation = (
                (rand_mask > mutation_bound[0]) & (
                    rand_mask < mutation_bound[1])
            ) * (2 * torch.rand(state1[key].shape) - 1)

            genes = genes1 + genes2 + genes_mutation
            new_genes[key] = genes

        child = Agent(self.height, self.width, self.time_limit)
        child.model.load_state_dict(new_genes)

    # Create the next generation
    def create_next_generation(self):
        self.breed(self.generation[0], self.generation[1])
