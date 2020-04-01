import random, math

import keras
import numpy as np
from Common.Brain import Brain
from Common.Memory import Memory, TreeMemory, VectorizedMemory
from map import Game, Scribe
import time
import threading
from Common.A2C import ActorCritic
import keras.backend as K
import tensorflow as tf
from BrainAnalizer import BrainAnalizer
import gym
from matplotlib import pyplot as PLT


class Agent:

    def __init__(self, brain, memory, environment):
        self.brain = brain
        self.memory = memory
        self.environment = environment
        self.total_scribe = Scribe(nr_of_quests=5, nr_of_stations=3, nr_of_locations=15 * 15, is_total=True)

        self.state_count = state_count
        self.batch_size = batch_size
        self.epsilon = epsilon_max
        self.discount = discount
        self.iteration = start_from_iteration
        self.iteration_limit = iteration_limit
        self.action_buffer = [action for action in range(self.environment.action_count)]


    def get_current_epsilon(self):
        epsilon = 0
        if self.iteration < self.iteration_limit:
            epsilon = 1 - 0.9 * (self.iteration / self.iteration_limit)
        else:
            epsilon = 0.001
        epsilones.append(epsilon)
        return epsilon

    def getAB(self):
        if self.iteration < learning_limit:
            current_progress = (self.iteration / learning_limit)
            return alpha_min + (alpha_max - alpha_min) * current_progress, beta_min + (beta_max - beta_min) * current_progress
        else:
            return alpha_max, beta_max

    def act(self, state):
        if random.random() > self.get_current_epsilon():
            state = self.encode_state(state)
            actions = self.brain.actor_model.predict([[state]])[0]
            action = np.random.choice(self.action_buffer, p=actions)
            return 0
        else:
            action = self.environment.sample_move()
            # action = self.environment.action_space.sample()
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
        new_state[-1] = np.clip(money / 2000, 0, 1)

        return new_state

    def run(self, should_replay=True, silent=True):
        state = self.environment.reset()
        total_reward = 0
        counter = np.zeros(6)
        # start = time.time()
        steps = max_steps
        for _ in range(max_steps):
            # print("taking step: ", _)
            action = self.act(state)
            next_state, reward, is_done, info = self.environment.step(action)
            total_reward += reward
            if is_done:
                # print("iteration ", agent.iteration, " is done")
                next_state = None
            self.observe_with_priority((state, action, reward, next_state, is_done))

            state = next_state
            if is_done:
                steps = _
                break
        if should_replay:
            alpha, beta = self.getAB()
            alphas.append(alpha)
            betas.append(beta)
            batch = memory.run_for_batch(alpha, beta)
            errors = brain.train(batch)
            tree_idx = batch[0]
            for i in range(len(errors)):
                memory.update_priority(tree_idx[i], errors[i])
            steps_buffer.append(steps)
            # self.total_scribe += self.environment.scribe
            # self.total_scribe += self.environment.scribe
        return total_reward


def init_memory(agent, nr_of_iter):
    while len(memory) < batch_size:
        agent.run(False)
    memory.batch_golden_retriever.start()
    for _ in range(nr_of_iter):
        agent.run(False)


def plot_figures():
    pass
    #
    # line1.set_ydata(grads_buffer)
    # line2.set_ydata(losses_buffer)
    # line3.set_ydata(rewards_buffer)
    # line4.set_ydata([item for item in zip(epsilones, alphas, betas)])
    # fig.canvas.draw()
    # PLT.draw()
    # PLT.show()



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
        if reward > max_reward:
            max_reward = reward
        rewards_buffer.append(reward)
        losses_buffer.append(agent.brain.current_loss)
        grads_buffer.append(agent.brain.current_grad)
        agent.iteration += 1
        print("Agent iteration:", agent.iteration)
        print("Current reward: ", reward)
        print("Best reward: {0:.4f}".format(max_reward))
        print(agent.environment.scribe)
        print("Current loss is equal to: ",agent.brain.current_loss)
        print("Average actor grad is equal to: ", agent.brain.current_grad)
        print("_____________________")
        if agent.iteration % synchronise_every == 0:
            save_to_json()
        if agent.iteration % plot_every == 0:
            plot_figures()



########################################################
test_run = False
check_policy = False
check_values = False
continue_learning = False
single_threading = True


#           H Y P E R            #
state_count = 37
batch_size = 2500
epsilon_max = 0.999
discount = 0.995
iteration_limit = 10000
learning_limit = 15000
start_from_iteration = 1
synchronise_every = 10
plot_every = 3
actor_learning_rate = 0.0002
critic_learning_rate = 0.00001
epsilones, alphas, betas = [], [], []

# sess = tf.Graph()
sess = tf.compat.v1.Session()
# K.set_session(sess)


#########################
#           MEMORY
fixed_batch_size = True
alpha_min = 0.4
alpha_max = 0.85
beta_min = 0.001
beta_max = 0.85
gamma = 2

########
grads_buffer = []
losses_buffer = []
rewards_buffer = []
steps_buffer = []

# fig = PLT.figure()
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
# PLT.ion()
#
# line1, = ax1.plot(grads_buffer)
# line2, = ax2.plot(losses_buffer)
# line3, = ax3.plot(rewards_buffer)
# line4, = ax4.plot([item for item in zip(epsilones, alphas, betas)])


environment = Game()
# environment = gym.make("CartPole-v1")
memory = VectorizedMemory(1500000, batch_size=batch_size, gamma=gamma, standarized_size=True)
# brain = ActorCritic(sess, environment.action_count, environment.state_count, tau=0.95)
brain = ActorCritic(sess, len(environment.action_space), state_count, tau=0.95, actor_learning_rate=actor_learning_rate,
                    critic_learning_rate=critic_learning_rate)
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
    while True:
        is_done = False
        state = environment.reset()
        total_reward = 0

        while not is_done:
            environment.render()

            action = np.argmax(brain.actor_model.predict([[state]]))
            next_state, reward, is_done, info = environment.step(action)
            total_reward += reward
            state = next_state
        brain.actor_model.load_weights("actor_model_weights.h5")
    brain.target_actor_model.load_weights("actor_model_weights.h5")
    brain.critic_model.load_weights("critic_model_weights.h5")
    brain.target_critic_model.load_weights("critic_model_weights.h5")
    brain_tester = BrainAnalizer(brain, environment.map_size)
    if check_policy:
        tanked_moves = brain_tester.check_policy(tanked=1, money=1)
        empty_moves = brain_tester.check_policy(tanked=0, money=1)
        print("Actions for tanked car: ")
        environment.print_map(tanked_moves)
        print("Actions for car without fuel: ")
        environment.print_map(empty_moves)
        print("map:")
        environment.print_map()
    if check_values:
        tanked_values = brain_tester.check_value(tanked=1, money=1)
        empty_values = brain_tester.check_value(tanked=0, money=1)
        print("quest values:")
        print(tanked_values[12][6])
        print(tanked_values[0][2])
        print(tanked_values[11][11])
        print(tanked_values[5][4])
        print(tanked_values[4][13])
        print("station values:")
        print(empty_values[7][7])
        print(empty_values[0][5])
        print(empty_values[10][6])
        print("all values:")
        for i in range(environment.map_size):
            line = ""
            for j in range(environment.map_size):
                line += "{:10}  ".format(str(tanked_values[i][j]))
            print(line)
