import random
import numpy as np
from Common.Brain import Brain
from Common.Memory import Memory
from CartPole.environment import Environment


class Agent:
    def __init__(self, brain, memory, state_count=4, batch_size=64, epsilon=0.99, discount=0.999, iteration_limit=500):
        self.brain = brain
        self.memory = memory
        self.state_count = state_count
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.discount = discount
        self.iteration = 0
        self.iteration_limit = iteration_limit

    def get_current_epsilon(self):
        if self.iteration < self.iteration_limit:
            return 1 - 0.9 * (self.iteration / self.iteration_limit)
        else:
            return 0.01

    def act(self, state):
        if random.random() > self.get_current_epsilon():
            action = np.argmax(self.brain.predict_one(state))
        else:
            action = random.getrandbits(1)
        return action

    def observe(self, sample):
        self.memory.add(sample)

    def replay(self):
        batch = self.memory.sample_batch(self.batch_size)

        empty_state = np.zeros(self.state_count)

        states = np.array([obs[0] for obs in batch])
        states_ = np.array([( empty_state if obs[3] is None else obs[3]) for obs in batch])

        p = self.brain.predict(states)
        p_ = self.brain.target_network.predict(states_)

        x = []
        y = []

        for i in range(len(batch)):
            sample = batch[i]
            state = sample[0]; action = int(sample[1]); reward = sample[2]; next_state = sample[3]
            target = p[i]
            if next_state is None:
                target[action] = reward
            else:
                target[action] = reward + self.discount * np.amax(p_[i])

            x.append(state)
            y.append(target)

        self.brain.train(x, y)


def init_memory(environment, agent, nr_of_iter):
    for _ in range(nr_of_iter):
        environment.run(agent, False)


environment = Environment()
memory = Memory(1000000)
brain = Brain(environment.state_count, environment.action_count)
agent = Agent(brain, memory)
max_reward = 0

init_memory(environment, agent, 20)


while True:
    reward = environment.run(agent)
    if reward > max_reward:
        max_reward = reward
    agent.iteration += 1
    if agent.iteration % 10 == 0:
        print("Agent iteration:", agent.iteration)
        print("Current reward: ", reward)
        print("Best reward: ", max_reward)
        print("_____________________")
