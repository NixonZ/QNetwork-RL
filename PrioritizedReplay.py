'''
https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
'''
from collections import namedtuple
from environment.env import device

class PrioritizedReplay:
    def __init__(self,buffer_size: int = 10e5, alpha = 0.5, beta = 0.5) -> None:
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta

        self.experience = namedtuple("Experience", 
            field_names=["state", "action", "reward", "next_state", "done"])
        self.data = namedtuple("Data", 
            field_names=["priority", "probability", "weight","index"])

        indexes = []
        datas = []
        for i in range(buffer_size):
            indexes.append(i)
            d = self.data(0,0,0,i)
            datas.append(d)

        # Pointers, using dictionaries.
        self.memory = {key: self.experience for key in indexes} # Actual buffer indexed by key
        self.memory_data = {key: data for key,data in zip(indexes, datas)} # Values for experience indexed by key
        pass

