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
        return epsilon

    def getAB(self):
        if self.iteration < learning_limit:
            current_progress = (self.iteration / learning_limit)
            return alpha_min + (alpha_max - alpha_min) * current_progress, beta_min + (beta_max - beta_min) * current_progress
        else:
            return alpha_max, beta_max

    def act(self, state):
        if act_stochaisticly:
            return self.act_stochaistic(state)
        else:
            if random.random() > self.get_current_epsilon():
                state = self.encode_state(state)
                action = np.max( self.brain.actor_model.predict([[state]]))

            else:
                action = self.environment.sample_move()
                # action = self.environment.action_space.sample()
            return action

    def act_stochaistic(self, state):
        state = self.encode_state(state)
        actions = self.brain.actor_model.predict([[state]])[0]
        action = np.random.choice(self.action_buffer, p=actions)
        return action, actions

    def observe_with_priority(self, sample):
        # max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        self.memory.add(sample)

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
        new_state[-1] = np.clip(money / 500, 0, 1)

        return new_state

    def run(self, should_replay=True, silent=True):
        state = self.environment.reset()
        total_reward = 0
        steps = max_steps
        for _ in range(max_steps):
            action, actions = self.act(state)
            next_state, reward, is_done, info = self.environment.step(action)
            total_reward += reward
            if is_done:
                next_state = None
            if act_stochaisticly:
                self.observe_with_priority((state, actions, reward, next_state, is_done))
            else:
                self.observe_with_priority((state, action, reward, next_state, is_done))
            state = next_state
            if is_done:
                steps = _
                break
        if should_replay:
            alpha, beta = self.getAB()
            batch = memory.run_for_batch(alpha, beta)
            errors = brain.train(batch)
            tree_idx = batch[0]
            # memory.gamma = np.average(errors)
            for i in range(len(errors)):
                memory.update_priority(tree_idx[i], errors[i])
            steps_buffer.append(steps)
        return total_reward


def init_memory(agent, nr_of_iter):
    while len(memory) < batch_size:
        agent.run(False)
    memory.batch_golden_retriever.start()
    for _ in range(nr_of_iter):
        agent.run(False)


def plot_axis(line, data, ax):
    line.set_ydata(data)
    line.set_xdata(range(len(data)))
    ax.set_xlim([0, len(data)])
    min = np.min(data)
    max = np.max(data)
    ax.set_ylim([min - np.abs(min) * 0.1, max + np.abs(max) * 0.1])




def plot_figures():
    for i, buffer in enumerate(small_buffers):
        big_buffers[i].append(np.average(buffer))
        buffer.clear()

    plot_axis(line1, big_buffers[0], ax1)
    plot_axis(line2, big_buffers[1], ax2)
    plot_axis(line3, big_buffers[2], ax3)
    plot_axis(line4, big_buffers[3], ax4)
    plot_axis(line5, big_buffers[4], ax5)
    plot_axis(line6, big_buffers[5], ax6)

    PLT.pause(5)





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
        IS_buffer.append(agent.brain.current_IS)
        invalid_buffer.append(agent.environment.scribe.invalid_action)
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
check_values = True
continue_learning = True
single_threading = True

act_stochaisticly = True


#           H Y P E R            #
state_count = 37
batch_size = 800
epsilon_max = 0.999
discount = 0.99
iteration_limit = 10000
learning_limit = 20000
start_from_iteration = 1
synchronise_every = 10
plot_every = 25
actor_learning_rate = 0.002
critic_learning_rate = 0.001

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
gamma = 16

########
grads_buffer, total_grads = [], []
losses_buffer, total_losses = [], []
rewards_buffer, total_rewards = [], []
steps_buffer, total_steps = [], []
IS_buffer, total_IS = [], []
invalid_buffer, total_invalid = [], []

small_buffers = [grads_buffer, losses_buffer, rewards_buffer, steps_buffer, IS_buffer, invalid_buffer]
big_buffers = [total_grads, total_losses, total_rewards, total_steps, total_IS, total_invalid]

fig = PLT.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)
PLT.ion()

line1, = ax1.plot(grads_buffer)
line2, = ax2.plot(losses_buffer)
line3, = ax3.plot(rewards_buffer)
line4, = ax4.plot(IS_buffer)
line5, = ax5.plot(steps_buffer)
line6, = ax6.plot(invalid_buffer)
PLT.show()




environment = Game()
# environment = gym.make("CartPole-v1")
memory = VectorizedMemory(1500000, batch_size=batch_size, gamma=gamma, standarized_size=True)
# brain = ActorCritic(sess, environment.action_count, environment.state_count, tau=0.95)
brain = ActorCritic(sess, len(environment.action_space), state_count, tau=0.9, actor_learning_rate=actor_learning_rate,
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
    # while True:
    #     is_done = False
    #     # state = environment.reset()
    #     total_reward = 0
    #
    #     while not is_done:
    #         # environment.render()
    #
    #         action = np.argmax(brain.actor_model.predict([[state]]))
    #         next_state, reward, is_done, info = environment.step(action)
    #         total_reward += reward
    #         state = next_state
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
        environment.print_map()

        print("all values:")
        for i in range(environment.map_size):
            line = ""
            for j in range(environment.map_size):
                line += "{:.2f}  ".format(tanked_values[i][j])
            print(line)
