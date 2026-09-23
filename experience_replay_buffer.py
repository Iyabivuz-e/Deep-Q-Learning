from collections import deque
import random

class ReplayMemory:
    def __init__(self, max_len, seed=None):
        ## Max Len is the mximum size of the memory
        self.memory = deque([], max_len)
        
        # Seed for reproductivity
        if seed is not None:
            random.seed(seed)
    # We append the transiton probabilities in the memory
    # transitions = (s, a, r, s')   
    def append_to_buffer(self, transitions):
        self.memory.append(transitions)
    
    # We take the random samples from the replay memory
    def sample_from_buffer(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    # We get the length of the memory
    def __len__(self):
        return len(self.memory)