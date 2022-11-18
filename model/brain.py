# **** We'll need a couple of things for this
# - We need a pytorch CNN model for the game board
# - We'll need to get the state from the snake
# - We'll need a way of mapping to the possible outputs
# - We'll need a way of evaluating the current score of the model

class Brain:
    def __init__(self, snake):
        self.snake = snake
