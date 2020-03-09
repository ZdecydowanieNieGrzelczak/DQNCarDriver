from queue import Queue


class HMM():
    def __init__(self, main_queue, timeout=200, filepath="database.txt", max_length=1000000):
        self.delay = timeout
        self.filepath = filepath
        self.queue = main_queue
        self.iterator = 0
        self.max_len = max_length




