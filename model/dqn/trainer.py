from collections import deque
import random
import torch
import copy

from model.dqn.model import DQNModel
import snake.utils as snake_utils
import model.utils as model_utils


class DQNTrainer:
    def __init__(self, snake, copy_timesteps=50, batch_size=32, buffer_length=128, alpha=1e-3, gamma=0.9, epsilon=1, epsilon_dec=0.9996, epsilon_end=0.01):
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

        self.train_step = 0

        # Store experience tuples (s_t, a_t, r_t+1, s_t+1)
        self.states = deque([], maxlen=buffer_length)
        self.actions = deque([], maxlen=buffer_length)
        self.rewards = deque([], maxlen=buffer_length)
        self.states_next = deque([], maxlen=buffer_length)

    # Save the current model
    def save_model(self, path):
        print(f"Saving model to '{path}'...")

        torch.save(self.model.state_dict(), path)

    # Update the target model with the current models weights
    def copy_to_target(self):
        self.target_model = copy.deepcopy(self.model)

    # Run a single game
    def train_step(self):
        # Run a step of the game based on epsilon greedy policy
        # Record the state, action, reward, and next state
        # If the amount of data is greater than or equal to the batch size:
        # - Select a random batch of experience tuples by index
        # - For each tuple, run the next state through the target model and select the top output from it
        # - Calculate loss and backpropagate
        # If x timesteps has proceeded, copy the current model to the target model
        # Decrease epsilon value
        # Update the training step

        # Run a single step from the game
        state = self.snake.get_state()

        if random.random() < self.epsilon:
            key = random.randint(0, 3)
        else:
            key, _ = model_utils.choose_key(state, self.model)

        reward = snake_utils.rewards[self.snake.update_state(key)]

        # **** WHAT AM I ACTUALLY BACKPROPOGATING - WE ONLY KEEP A SINGLE VALUE ???

        pass
