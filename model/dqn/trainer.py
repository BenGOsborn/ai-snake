from model.ga.model import Model
from model.agent import Agent


class DQNTrainer:
    def __init__(self, epsilon=0.9, epsilon_dec=0.996, epsilon_end=0.01, time_limit=1000):
        self.time_limit = time_limit

        # Store experience tuples (s_t, a_t, r_t+1, s_t+1)
        self.states = None
        self.actions = None
        self.rewards = None

    def run_epoch(self):
        pass
