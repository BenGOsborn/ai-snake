from model.agent import Agent


class Trainer:
    def __init__(self, height, width, generation_size):
        self.generation_size = generation_size

        self.generation = [
            Agent(height, width) for _ in range(generation_size)
        ]

    # Evaluate all agents in the current population
    def evaluate_population(self):
        for elem in self.generation:
            elem.evaluate()
