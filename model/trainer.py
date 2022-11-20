import torch

from model.model import Model
from model.agent import Agent


class Trainer:
    def __init__(self, snake, generation_size, mutation_chance):
        self.generation_size = generation_size
        self.mutation_chance = mutation_chance

        self.snake = snake

        # Keep track of the best agent
        self.best_fitness = -torch.inf
        self.best_agent = None

        # Initialize generation
        self.generation = [
            Agent(self.snake, Model().eval()) for _ in range(generation_size)
        ]

    # Evaluate all agents in the current population and get the current average and max fitness
    def evaluate_population(self):
        for agent in self.generation:
            agent.evaluate()

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
        model = Model()
        model.load_state_dict(new_genes)

        return Agent(self.snake, model.eval())

    # Save the highest fitness agent
    def save_best_agent(self, path):
        torch.save(self.best_agent.model.state_dict(), path)

    # Create the next generation
    def create_next_generation(self):
        # Select the best performing agents
        fitness = torch.tensor(
            [agent.fitness for agent in self.generation],
            dtype=torch.float
        )

        # Update the best agent
        argmax = torch.argmax(fitness)
        if fitness[argmax] > self.best_fitness:
            self.best_fitness = fitness[argmax]
            self.best_agent = self.generation[argmax]

        # Breed fit agents to create new generation
        probs = torch.softmax(fitness, dim=0)
        distribution = torch.distributions.categorical.Categorical(probs=probs)

        new_generation = []

        for _ in range(len(self.generation)):
            parent1, parent2 = distribution.sample((2,)).tolist()

            child = self.breed(
                self.generation[parent1],
                self.generation[parent2]
            )
            new_generation.append(child)

        # Replace old generation
        self.generation = new_generation
