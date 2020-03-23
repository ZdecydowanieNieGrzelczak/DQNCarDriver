import random, math

import keras
import numpy as np
from Common.Brain import Brain
from Common.Memory import Memory, TreeMemory, VectorizedMemory
from map import Game
import time
import threading
from Common.A2C import ActorCritic
import keras.backend as K
import tensorflow as tf


class Agent:
    def __init__(self, brain, memory, environment):
        self.brain = brain
        self.memory = memory
        self.environment = environment

        self.state_count = state_count
        self.batch_size = batch_size
        self.epsilon = epsilon_max
        self.discount = discount
        self.iteration = start_from_iteration
        self.iteration_limit = iteration_limit

    def get_current_epsilon(self):
        if self.iteration < self.iteration_limit:
            return 1 - 0.9 * (self.iteration / self.iteration_limit)
        else:
            return 0.001

    def getAB(self):
        if self.iteration < self.iteration_limit:
            current_progress = (self.iteration / self.iteration_limit)
            return alpha_min + (alpha_max - alpha_min) * current_progress, beta_min + (beta_max - beta_min) * current_progress
        else:
            return alpha_max, beta_max

    def act(self, state):
        if random.random() > self.get_current_epsilon():
            state = self.encode_state(state)
            action = np.argmax(self.brain.actor_model.predict([[state]]))
        else:
            action = self.environment.sample_move()
        return action


    def observe_with_priority(self, sample):
        # max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        self.memory.add(sample)

    def replay(self):
        batch = self.memory.sample_batch(self.batch_size)

        empty_state = np.zeros(self.state_count)

        states = np.array([self.encode_state(obs[0]) for obs in batch])
        states_ = np.array([(empty_state if obs[3] is None else self.encode_state(obs[3])) for obs in batch])

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

            state = self.encode_state(state)
            x.append(state)
            y.append(target)

        self.brain.train(x, y)

    def encode_state(self, state):
        map_size = 15
        quest_number = 5

        pos = state[0]
        cargo = state[1]
        gas = state[2]
        money = state[3]
        new_state = np.zeros(map_size * 2 + quest_number + 2, )
        new_state[pos[0]] = 1
        new_state[map_size + pos[1]] = 1
        for i in range(len(cargo)):
            new_state[map_size * 2 + i] = cargo[i]
        new_state[-2] = gas / 500
        new_state[-1] = np.min(money / 2000, 1)

        return new_state

    def run(self, should_replay=True, silent=True):
        state = self.environment.reset()
        total_reward = 0
        counter = np.zeros(6)
        # start = time.time()
        for _ in range(max_steps):
            # print("taking step: ", _)
            action = self.act(state)
            if not silent:
                counter[action] += 1
            next_state, reward, is_done, info = self.environment.step(action)
            total_reward += reward
            if is_done:
                next_state = None
            self.observe_with_priority((state, action, reward, next_state, is_done))

            state = next_state
            if is_done:
                break
        if should_replay:
            alpha, beta = self.getAB()
            batch = memory.run_for_batch(alpha, beta)
            errors = brain.train(batch)
            tree_idx = batch[0]
            for i in range(len(errors)):
                memory.update_priority(tree_idx[i], errors[i])
        if not silent:
            print("Counter : ", counter)
        return total_reward


def init_memory(agent, nr_of_iter):
    while len(memory) < batch_size:
        agent.run(False)
    memory.batch_golden_retriever.start()
    for _ in range(nr_of_iter):
        agent.run(False)




def save_to_json():
    model_jason = brain.actor_model.to_json()

    with open("actor.json", "w") as json_file:
        json_file.write(model_jason)
    brain.actor_model.save_weights("actor_model_weights.h5")
    model_jason = brain.critic_model.to_json()

    with open("critic.json", "w") as json_file:
        json_file.write(model_jason)
    brain.critic_model.save_weights("critic_model_weights.h5")


def run_agent():
    max_reward = -10000000000000000
    best_out_of_ten = -10000000000000000
    stats = [0, 0, 0]
    while True:
        reward = agent.run(should_replay=single_threading)
        if agent.environment.has_tanked:
            stats[0] += 1
        if agent.environment.started:
            stats[1] += 1
        if agent.environment.ended:
            stats[2] += 1
        if reward > max_reward:
            max_reward = reward
        if reward > best_out_of_ten:
            best_out_of_ten = reward
        agent.iteration += 1

        if agent.iteration % synchronise_every == 0:
            # agent.brain.update_model()
            # if not single_threading:
            #     agent.brain.model.set_weights(synch_model.get_weights())
            # agent.brain.update_target_network()
            print("Agent iteration:", agent.iteration)
            print("Current reward: ", reward)
            print("Best reward: ", math.floor(max_reward))
            print("Best of this episode: ", math.floor(best_out_of_ten))
            # print("Len of memory: ", len(memory))
            print("The questes: T:", stats[0], " P:", stats[1], " C:", stats[2], " out of: ", synchronise_every )
            print("_____________________")
            stats = [0, 0, 0]
            best_out_of_ten = -10000
            if all_loaded or single_threading:
                save_to_json()



########################################################
test_run = False
continue_learning = False
single_threading = True


#           H Y P E R            #
state_count = 37
batch_size = 200
epsilon_max = 0.99
discount = 0.985
iteration_limit = 1000
start_from_iteration = 0
synchronise_every = 5

# sess = tf.Graph()
sess = tf.compat.v1.Session()
# K.set_session(sess)


#########################
#           MEMORY
fixed_batch_size = True
alpha_min = 0.4
alpha_max = 0.9
beta_min = 0.001
beta_max = 0.9
gamma = 4

environment = Game()
memory = VectorizedMemory(1000000, batch_size=batch_size, gamma=gamma, standarized_size=False)
learning_rate = 0.002
brain = ActorCritic(sess, environment.action_count, environment.state_count, tau=0.95)
agent = Agent(brain, memory, environment)
max_steps = 15000
max_reward = -100000
best_out_of_ten = -10000
all_loaded = False


if continue_learning:
    brain.actor_model.load_weights("actor_model_weights.h5")
    brain.target_actor_model.load_weights("actor_model_weights.h5")
    brain.critic_model.load_weights("critic_model_weights.h5")
    brain.target_critic_model.load_weights("critic_model_weights.h5")
    print("loaded")






#   THREADS #
if not single_threading:
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    synch_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    synch_model.load_weights("model_weights.h5")


    learning_thread = threading.Thread(target=learn_and_sync, daemon=True)
    # # playing_thread = threading.Thread(target=run_agent, daemon=True)
    learning_thread.start()



if not test_run:
    init_memory(agent, 20)
    run_agent()
else:
    environment = Game()
    # brain = Brain(environment.state_count, environment.action_count, learning_rate=learning_rate)
    brain.actor_model.load_weights("actor_model_weights.h5")
    brain.target_actor_model.load_weights("actor_model_weights.h5")
    brain.critic_model.load_weights("critic_model_weights.h5")
    brain.target_critic_model.load_weights("critic_model_weights.h5")
    while True:
        reward = agent.run(False, False)
        print(reward)