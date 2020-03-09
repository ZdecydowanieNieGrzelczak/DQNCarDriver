import random
import time


class Memory:
    """
    Simple ring buffer for storage of data that will be used while "replaying"
    """

    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)

        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, item):
        return self.data[(self.start + item) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield(self[i])

    def sample_batch(self, size):
        random.seed(time.time())
        bias = len(self)
        result = []
        for i in range(size):
            random_index = random.randint(0, bias - 1)
            result.append(self[random_index])
        return result

    def add(self, element):
        self.append(element)