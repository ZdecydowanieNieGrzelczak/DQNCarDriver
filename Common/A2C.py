import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque


def encode_state(state):
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


class ActorCritic():
    current_loss = 0
    current_grad = 0
    current_IS = 0

    def __init__(self, sess, action_space, observation_space, actor_learning_rate=0.0001, critic_learning_rate=0.001,
                 discount=0.999, tau=0.95, hidden=(25, 40, 15)):
        tf.compat.v1.disable_eager_execution()

        self.sess = sess
        self.hidden = hidden
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount = discount
        self.tau = tau
        self.observation_space = observation_space
        self.action_space = action_space
        self.empty_action = np.zeros(shape=action_space)
        self.action_placeholder = K.placeholder(self.action_space, dtype=tf.float64)

        self.actor_state_input, self.actor_model = self.create_actor_model(self.hidden)
        _, self.target_actor_model = self.create_actor_model(self.hidden)
        self.advantage_placeholder = K.placeholder(1, dtype=tf.float64)

        self.actor_critic_grad = K.placeholder([None, self.action_space], dtype=tf.float32)
        self.actor_loss = -tf.math.log(self.action_placeholder) * self.advantage_placeholder

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.keras.optimizers.Adam(self.actor_learning_rate).apply_gradients(grads)


        self.critic_input, self.critic_model = self.create_advantage_critic_model((25, 15))

        _, self.target_critic_model = self.create_advantage_critic_model((25, 15))

        self.actor_advantage_grads = tf.gradients(self.actor_loss, self.action_placeholder)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def create_actor_model(self, hidden):

        state_input = Input(shape=(self.observation_space,), name="actor_inp")
        h1 = Dense(hidden[0], activation="relu")(state_input)
        # h2 = Dense(hidden[1], activation="relu")(h1)
        h3 = Dense(hidden[2], activation="relu")(h1)

        output = Dense(self.action_space, activation="softmax")(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(learning_rate=self.actor_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self, hidden_env, hidden_action, hidden_merged):
        state_input = Input(shape=(self.observation_space,), name="critic_input")
        state_h1 = Dense(hidden_env[0], activation='relu')(state_input)
        state_h2 = Dense(hidden_env[1], name="state_h2")(state_h1)

        action_input = Input(shape=(self.action_space,), name="action_inpt")
        action_h1 = Dense(hidden_action, name="action_h1")(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(hidden_merged, name="merged", activation='relu')(merged)
        output = Dense(units=1, name="output_crit", activation='linear')(merged_h1)
        model = Model(input=[state_input, action_input],
                      output=output)

        adam = Adam(learning_rate=self.critic_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def create_advantage_critic_model(self, hidden_size):
        critic_input = Input(shape=(self.observation_space,), name="critic_input")
        critic_h1 = Dense(units=hidden_size[0], activation="relu")(critic_input)
        critic_h2 = Dense(units=hidden_size[1], activation="relu")(critic_h1)
        output = Dense(units=1, name="crit_output", activation="linear")(critic_h2)
        model = Model(input=critic_input, output=output)

        adam = Adam(learning_rate=self.critic_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return critic_input, model

    def train(self, batch):
        TD_errors = self._advantage_train(batch)
        # self._train_actor(batch)
        self.update_targets()
        return np.abs(TD_errors)

    def _advantage_train(self, batch):
        rewards, states, actions = [], [], []
        batch_idx, samples, ISWeights = batch
        TD_errors = []

        for i, sample in enumerate(samples):
            cur_state, action, reward, new_state, done = sample
            action_taken = action
            current_state = self.target_critic_model.predict([[np.array(encode_state(cur_state))]])[0][0]
            TD_error = reward - current_state
            if not done:
                future_state = self.target_critic_model.predict([[np.array((encode_state(new_state)))]])[0][0]
                reward += self.discount * future_state
                TD_error += self.discount * future_state
            rewards.append(reward)
            states.append(encode_state(cur_state))
            TD_errors.append(TD_error)

            grad = self.sess.run(self.actor_advantage_grads, feed_dict= {
                self.action_placeholder: action_taken,
                self.advantage_placeholder: [TD_error],
            })[0]

            weighted_grad = grad * ISWeights[i]


            self.sess.run(self.optimize, feed_dict={
                self.actor_critic_grad: [weighted_grad],
                self.actor_state_input: [encode_state(cur_state)]
            })

        history = self.critic_model.fit([states], rewards, epochs=2, verbose=0, batch_size=len(samples),
                                        sample_weight=ISWeights)

        self.current_loss = np.average(history.history['loss'])
        self.current_grad = np.average(TD_errors)
        self.current_IS = np.average(ISWeights)
        return TD_errors

    def update_targets(self):
        self._update_actor_target()
        self._update_critic_target()

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = self.tau * actor_target_weights[i] + actor_model_weights[i] * (1 - self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = self.tau * critic_target_weights[i] + critic_model_weights[i] * (1 - self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def copy_targets(self):
        self.target_critic_model.set_weights(self.critic_model.get_weights())
        self.target_actor_model.set_weights(self.actor_model.get_weights())
