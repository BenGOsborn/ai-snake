import torch
from multiprocessing.dummy import Pool as ThreadPool

from model.agent import Agent


class Trainer:
    def __init__(self, height, width, generation_size, mutation_chance, evaluations, time_limit):
        self.height = height
        self.width = width
        self.generation_size = generation_size
        self.mutation_chance = mutation_chance
        self.evaluations = evaluations
        self.time_limit = time_limit

        # Keep track of the best agent
        self.best_fitness = -1
        self.best_agent = None

        # Initialize generation
        self.generation = [
            Agent(height, width, evaluations, time_limit) for _ in range(generation_size)
        ]

    # Evaluate all agents in the current population
    def evaluate_population(self):
        pool = ThreadPool(32)
        pool.map(lambda x: x.evaluate(), self.generation)

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
            genes2 = (rand_mask > mutation_bound[1]) * state2[key]
            genes_mutation = (
                (rand_mask > mutation_bound[0]) & (
                    rand_mask <= mutation_bound[1])
            ) * (2 * torch.rand(state1[key].shape) - 1)

            genes = genes1 + genes2 + genes_mutation
            new_genes[key] = genes

        # Create new child with new genes
        child = Agent(
            self.height,
            self.width,
            self.evaluations,
            self.time_limit
        )
        child.model.load_state_dict(new_genes)

        return child

    # Save the highest fitness agent
    def save_best_agent(self, path):
        torch.save(self.best_agent.model.state_dict(), path)

    # Create the next generation
    def create_next_generation(self):
        # Select from distribution based on fitness
        fitness = torch.tensor(
            [agent.fitness for agent in self.generation],
            dtype=torch.float
        )
        probs = torch.softmax(fitness, dim=0)
        distribution = torch.distributions.categorical.Categorical(probs=probs)

        # Update the best agent
        argmax = torch.argmax(fitness)
        if fitness[argmax] > self.best_fitness:
            self.best_fitness = fitness[argmax]
            self.best_agent = self.generation[argmax]

        # Breed fit agents to create new generation
        new_generation = []
        for _ in range(self.generation_size):
            parent1, parent2 = distribution.sample((2,)).tolist()

            child = self.breed(
                self.generation[parent1],
                self.generation[parent2]
            )
            new_generation.append(child)

        self.generation = new_generation
