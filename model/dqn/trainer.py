import torch
import copy

from model.dqn.model import DQNModel


class DQNTrainer:
    def __init__(self, snake, copy_timesteps=50, batch_size=32, alpha=1e-3, gamma=0.9, epsilon=1, epsilon_dec=0.9996, epsilon_end=0.01):
        self.copy_timesteps = copy_timesteps
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end

        self.snake = snake

        # Initialize model
        self.model = DQNModel()
        self.target_model = copy.deepcopy(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

        # Store experience tuples (s_t, a_t, r_t+1, s_t+1)
        self.states = None
        self.actions = None
        self.rewards = None
        self.states_next = None

    # Save the current model
    def save_model(self, path):
        print(f"Saving model to '{path}'...")

        torch.save(self.model.state_dict(), path)

    # Update the target model with the current models weights
    def copy_to_target(self):
        self.target_model = copy.deepcopy(self.model)

    # Run a single game
    def train_step(self):
        pass
