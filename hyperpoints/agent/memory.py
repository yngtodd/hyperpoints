import random
from collections import namedtuple


transition = namedtuple(
  'transition',
  ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __repr__(self):
        return f'Replay memory - Capacity: {self.capacity}'

    def __len__(self):
        return len(self.memory)
