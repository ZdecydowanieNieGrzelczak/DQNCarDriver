import gym
import numpy as np
import random
from map import Game

def frozen_lake():
    env = gym.make('FrozenLake-v0')

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    return
    learning_rate = 0.8
    gamma = 0.95
    num_episodes = 5000
    rewards = []
    for i in range(num_episodes):

        state = env.reset()
        total_reward = 0
        is_done = False
        while not is_done:
            action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            next_state, reward, is_done, _ = env.step(action)

            Q[state, action] = Q[state, action] + learning_rate * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        if i % 100:
            print("Iteration nr: ", i)


def blackjack_model():
    pass


def blackjack(value_pair):

    env = gym.make('Blackjack-v0')

    nr_of_episodes = 200000
    gamma = value_pair[0]
    learning_rate = value_pair[1]

    Q = np.zeros([env.observation_space[0].n, env.observation_space[1].n, env.action_space.n])
    rewards = []

    for i in range(nr_of_episodes):
        reward = 0
        state = env.reset()
        is_done = state[2]
        while is_done:
            is_done = env.reset()[2]

        epsilon = 0.1 + 1 * (i / 5000.0)
        while not is_done:

            if random.random() > epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state[0], state[1], :])

            next_state, reward, is_done, _ = env.step(action)

            Q[state[0], state[1], action] = Q[state[0], state[1], action] + learning_rate * (reward + gamma * np.max(Q[next_state[0],
                                                                 next_state[1], :]) - Q[state[0], state[1], action])

            state = next_state


        rewards.append(reward)


    # print(rewards)

    rewards = []
    for i in range(nr_of_episodes):
        state = env.reset()
        is_done = state[2]
        reward = 0
        while is_done:
            is_done = env.reset()[2]

        while not is_done:
            action = np.argmax(Q[state[0], state[1], :])

            next_state, reward, is_done, _ = env.step(action)
            state = next_state

        rewards.append(reward)

    # return rewards.count(1)
    # print(Q[20, 10, :])
    print("Wins: ", rewards.count(1))
    print("Loses: ", rewards.count(-1))


best_values = [0.99, 0.01]

#
# game = Game()
#
# game.print_map()



x, y = zip(best_values)

print(x)
print(y)