from abc import ABC

import torch as T


class Agent(ABC):
    def __init__(self):
        return

    def choose_action(self, state):
        return

    def set_target(self, target, message):
        self.message = T.tensor(message)

    def train(self):
        return

    def eval(self):
        return
