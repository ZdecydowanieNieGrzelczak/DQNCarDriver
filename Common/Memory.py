import random
import time
from queue import Queue

import numpy as np
import copy
import threading
from operator import itemgetter




class VectorizedMemory:
    min_priority = 0.0001
    max_priority = 1
    start = 0
    end = 0
    random.seed(time.time())
    next_batch = None
    is_ready = False
    alpha = 0
    beta = 0
    working_data = None
    working_priority = None
    delay = 0.005
    large_delay = 0.02
    batch_queue = Queue(5)
    _datalock = False
    working_alpha = 0
    working_beta = 0

    def __init__(self, size, use_slices=True, batch_size=2048, gamma=4):
        self.data = [None] * (size + 1)
        self.priority_buffer = np.zeros(shape=(size + 1))
        self.use_slices = use_slices
        self.batch_golden_retriever = threading.Thread(target=self.batch_preparer, daemon=True)
        self.batch_size = batch_size
        self.gamma = gamma

    def append(self, element):
        while(self._datalock):
            time.sleep(self.delay)
        self.data[self.end] = element
        self.priority_buffer[self.end] = self.max_priority
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
            yield self[i]

    def add(self, element):
        self.append(element)

    def update_priority(self, index, error):
        priority = np.clip(error / self.gamma, self.min_priority, self.max_priority)
        self.priority_buffer[index] = priority

    def batch_preparer(self):
        while True:
            if self.batch_queue.full():
                time.sleep(self.large_delay)
            else:
                self._datalock = True
                if self.end > self.start:
                    self.working_data = copy.deepcopy(self.data[:self.end])
                    self.working_priority = copy.deepcopy(self.priority_buffer[:self.end])
                else:
                    self.working_data = copy.deepcopy(self.data)
                    self.working_priority = copy.deepcopy(self.priority_buffer)
                self._datalock = False
                self.working_alpha = self.alpha
                self.working_beta = self.beta
                self.sample_with_priority()

    def sample_with_priority(self):
        indexes, ISWeights = self.vectorized(self.working_priority)
        next_data = list(itemgetter(*indexes)(self.working_data))
        next_batch = [indexes, next_data, ISWeights]
        self.batch_queue.put(next_batch)


    def vectorized(self, prob_matrix):

        s = np.float_power(prob_matrix, self.working_alpha)
        r = np.random.rand(prob_matrix.shape[0])
        k = np.where(s > r)
        probabilities = s[k]
        while np.shape(k)[1] < self.batch_size:
            s[k] = 1
            r = np.random.rand(prob_matrix.shape[0])
            k2 = np.where(s > r)
            np.append(probabilities, s[k2])
            np.append(k, k2)

        if np.shape(k)[1] > self.batch_size:
            difference = np.shape(k)[1] - self.batch_size
            if self.use_slices:
                indices = np.arange(np.shape(k)[1])
                np.random.shuffle(indices)
                k = k[0][indices]
                probabilities = probabilities[indices]
                k = k[:-difference]
                probabilities = probabilities[:-difference]

            else:
                new_k = np.asarray(k)
                k = np.delete(k, (np.random.choice(new_k, difference, replace=False)))

        ISWeights = np.float_power(((1 / len(self)) * (1 / probabilities)), self.working_beta)
        k = k.tolist()

        return k, ISWeights

    def run_for_batch(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        while self.batch_queue.empty():
            time.sleep(self.delay)
        batch = self.batch_queue.get()
        return batch



#####################################################################################
#                                                                                   #
#                                                                                   #
#                                                                                   #
#                                      DEPRECIATED                                  #
#                                                                                   #
#                                                                                   #
#                                                                                   #
#####################################################################################

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


class TreeMemory(object):
    epsilon = 0.001  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 1e-4  # annealing the bias
    abs_err_upper = 8   # for stability refer to paper

    def __init__(self, size):
        self.tree = SumTree(size)

    def store(self, error, transition):
        p = self._get_priority(error)
        self.tree.add_new_priority(p, transition)

    def sample_batch(self, n):
        random.seed(time.time())
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority + self.epsilon
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(float(self.tree.capacity * prob))
            batch_idx.append(idx)
            batch_memory.append(data)

        # ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return (batch_idx, batch_memory, ISWeights)

    def update_whole_list(self, indexes, errors):
        for idx, error in indexes, errors:
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error = np.abs(error)
        error += self.epsilon  # avoid 0
        clipped_error = min(error, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)


class SumTree(object):

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        # leaf_idx = self.data_pointer
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data  # update data_frame
        self.update(leaf_idx, p)  # update tree_frame
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):  # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound - self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]  # the root
