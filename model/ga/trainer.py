import torch

from model.ga.model import GAModel
from model.agent import Agent


class GATrainer:
    def __init__(self, snake, generation_size, top_k, mutation_chance, mutation_pop_chance):
        self.generation_size = generation_size
        self.top_k = top_k
        self.mutation_chance = mutation_chance
        self.mutation_pop_chance = mutation_pop_chance

        self.snake = snake

        # Initialize generation
        self.generation = [
            Agent(self.snake, GAModel().eval()) for _ in range(generation_size)
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
            rand1 = torch.rand(state[key].shape)
            rand2 = 2 * torch.rand(state[key].shape) - 1

            original = (rand1 >= self.mutation_chance) * state[key]
            mutated = (rand1 < self.mutation_chance) * rand2

            genes = original + mutated
            new_genes[key] = genes

        # Create new child with new genes
        model = GAModel()
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

        # Select top K elements
        values, indices = fitness.topk(self.top_k)

        # Update the best agent
        if values[0] > self.best_fitness:
            print("NEW BEST")
            self.best_fitness = values[0]
            self.best_agent = self.generation[indices[0]]

        # Mutate agents to create better ones
        probs = torch.softmax(values, dim=0)
        distribution = torch.distributions.categorical.Categorical(probs=probs)

        new_generation = []

        while len(new_generation) < int(len(self.generation) * (1 - self.mutation_pop_chance)):
            # Add original genes
            original_index = distribution.sample().item()
            original = self.generation[indices[original_index]]

            new_generation.append(original)

            # Add mutated genes
            mutated = self.mutate(original)
            new_generation.append(mutated)

        for _ in range(len(self.generation) - len(new_generation)):
            # Add random genes to the pool
            rand = Agent(self.snake, GAModel().eval())
            new_generation.append(rand)

        # Replace old generation
        self.generation = new_generation
