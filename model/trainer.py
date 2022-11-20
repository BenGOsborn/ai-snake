import torch

from model.model import Model
from model.agent import Agent


class Trainer:
    def __init__(self, snake, generation_size, mutation_chance):
        self.generation_size = generation_size
        self.mutation_chance = mutation_chance

        self.snake = snake

        # Initialize generation
        self.generation = [
            Agent(self.snake, Model().eval()) for _ in range(generation_size)
        ]

        # Keep track of the best agent
        self.best_fitness = -torch.inf
        self.best_agent = None

    # Evaluate all agents in the current population and get the current average and max fitness
    def evaluate_population(self):
        for agent in self.generation:
            agent.evaluate()

    # Mutate an agent
    def mutate(self, agent):
        state = agent.model.state_dict()

        # Merge genes together randomly
        new_genes = {}

        # Mutate all genes
        for key in state:
            rand1 = 2 * torch.rand(state[key].shape) - 1
            rand2 = 2 * torch.rand(state[key].shape) - 1

            original = (rand1 >= self.mutation_chance) * state[key]
            mutated = (rand1 < self.mutation_chance) * rand2

            genes = original + mutated
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

        print(f"Mean: {torch.mean(fitness)} - Max: {torch.max(fitness)}")

        # Update the best agent
        argmax = torch.argmax(fitness)
        if fitness[argmax] > self.best_fitness:
            print("NEW BEST")
            self.best_fitness = fitness[argmax]
            self.best_agent = self.generation[argmax]

        # Mutate agents to create better ones
        probs = torch.softmax(fitness, dim=0)
        distribution = torch.distributions.categorical.Categorical(probs=probs)

        new_generation = []

        while len(new_generation) < int(len(self.generation) * (1 - self.mutation_chance)):
            # Add original genes
            original = distribution.sample().item()
            new_generation.append(self.generation[original])

            # Add mutated genes
            mutated = self.mutate(self.generation[original])
            new_generation.append(mutated)

        for _ in range(len(self.generation) - len(new_generation)):
            # Add random genes to the pool
            rand = Agent(self.snake, Model().eval())
            new_generation.append(rand)

        # Replace old generation
        self.generation = new_generation
