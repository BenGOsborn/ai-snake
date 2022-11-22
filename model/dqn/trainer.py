from collections import deque
import random
import torch
import copy

from model.dqn.model import DQNModel
import snake.utils as snake_utils
import model.utils as model_utils


class DQNTrainer:
    def __init__(self, snake, copy_steps=50, batch_size=64, buffer_length=1000, alpha=5e-4, gamma=0.99, epsilon=1, epsilon_dec=0.9996, epsilon_min=0.01):
        self.copy_steps = copy_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min

        self.snake = snake

        # Initialize model
        self.model = DQNModel()
        self.target_model = copy.deepcopy(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

        self.step = 0

        # Store experience tuples (s_t, a_t, r_t+1, s_t+1)
        self.states = deque([], maxlen=buffer_length)
        self.actions = deque([], maxlen=buffer_length)
        self.rewards = deque([], maxlen=buffer_length)
        self.new_states = deque([], maxlen=buffer_length)

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
            action = random.randint(0, 3)
        else:
            action, _ = model_utils.choose_key(state, self.model)

        reward = snake_utils.rewards[self.snake.update_state(action)]

        new_state = self.snake.get_state()

        # Update state data
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)

        # Train a batch
        if len(self.states) >= self.batch_size:
            indices = [i for i in range(len(self.states))]
            batch = random.sample(indices, self.batch_size)

            # Create predictions of Q values for the current state
            states = torch.tensor([self.states[i] for i in batch])

            print(states)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec

        # Update training step
        self.step += 1

        # Copy the target model to the current model
        if self.step % self.copy_steps == 0:
            pass
