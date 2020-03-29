import numpy as np

class BrainAnalizer:

    def __init__(self, brain, map_size):
        self.brain = brain
        self.map_size = map_size
        self.state_size = brain.observation_space
        self.action_count = brain.action_space

    def check_policy(self, tanked=1, money=1, cargo=(0, 0, 0, 0, 0)):
        action_matrix = np.zeros(shape=(self.map_size, self.map_size))

        for i in range(self.map_size):
            for j in range(self.map_size):
                action_matrix[i][j] = np.argmax(self.brain.actor_model.predict([[self.encode_state([i, j], cargo, tanked, money,)]]))
        return action_matrix

    def check_value(self, tanked=1, money=1, cargo=(0, 0, 0, 0, 0)):
        values_matrix = np.zeros(shape=(self.map_size, self.map_size))
        action_array = np.zeros(shape=(self.action_count,))
        for i in range(self.map_size):
            for j in range(self.map_size):
                sum = 0
                for x in range(self.action_count):
                    action_array[x] = 1
                    sum += self.brain.critic_model.predict([np.array(([self.encode_state([i, j], cargo, tanked, money, )])), np.array([action_array])])
                    action_array[x] = 0
                values_matrix[i][j] = sum / self.action_count
        return values_matrix

    def encode_state(self, pos, cargo, gas, money):
        state = np.zeros(shape=(self.state_size,))
        state[pos[0]] = 1
        state[self.map_size + pos[1]] = 1
        for i in range(len(cargo)):
            state[self.map_size * 2 + i] = cargo[i]
        state[-2] = gas
        state[-1] = money
        return state