import torch

from model.model import Model
from model.agent import Agent


class Trainer:
    def __init__(self, snake, generation_size, top_agents, mutation_chance, evaluations, time_limit, stuck_limit):
        self.generation_size = generation_size
        self.top_agents = top_agents
        self.mutation_chance = mutation_chance
        self.evaluations = evaluations
        self.time_limit = time_limit
        self.stuck_limit = stuck_limit

        self.snake = snake

        # Keep track of the best agent
        self.best_fitness = -1
        self.best_agent = None

        # Initialize generation
        self.generation = [
            Agent(self.snake, Model(), self.evaluations, time_limit, stuck_limit) for _ in range(generation_size)
        ]

    # Evaluate all agents in the current population and get the current average and max fitness
    def evaluate_population(self):
        for agent in self.generation:
            agent.evaluate()

        fitness = torch.tensor(
            [elem.fitness for elem in self.generation],
            dtype=torch.float
        )

        return torch.mean(fitness), torch.max(fitness)

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

        child = Agent(
            self.snake,
            model,
            self.evaluations,
            self.time_limit,
            self.stuck_limit,
        )

        return child

    # Save the highest fitness agent
    def save_best_agent(self, path):
        torch.save(self.best_agent.model.state_dict(), path)

    # Create the next generation
    def create_next_generation(self):
        # Select the best performing agents
        fitness = sorted(
            [(i, agent.fitness) for i, agent in enumerate(self.generation)],
            reverse=True,
            key=lambda x: x[1]
        )[:self.top_agents]

        values = torch.tensor([elem[1] for elem in fitness])
        indices = torch.tensor([elem[0] for elem in fitness])

        # Update the best agent
        if values[0] > self.best_fitness:
            self.best_fitness = values[0]
            self.best_agent = self.generation[indices[0]]

        # Breed fit agents to create new generation
        probs = torch.softmax(values, dim=0)
        distribution = torch.distributions.categorical.Categorical(probs=probs)

        new_generation = []
        for _ in range(self.generation_size):
            parent1, parent2 = distribution.sample((2,)).tolist()

            child = self.breed(
                self.generation[indices[parent1]],
                self.generation[indices[parent2]]
            )
            new_generation.append(child)

        self.generation = new_generation
