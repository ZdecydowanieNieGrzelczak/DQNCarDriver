import keras
import numpy as np


class Brain:
    def __init__(self, state_count, action_count, size_of_hidden=(128, 64, 32), learning_rate=0.00002):
        self.state_count = state_count
        self.action_count = action_count
        self.size_of_hidden = size_of_hidden
        self.model = self.create_model()
        self.target_network = self.create_model()
        self.learning_rate = learning_rate
        self.async_model = self.create_model()

    def create_model(self):
        model = keras.Sequential()
        # model.add(keras.layers.Dense(activation='relu', input_dim=self.state_count, units=self.size_of_hidden[0]))
        model.add(keras.layers.Dense(activation='relu', input_dim=self.state_count, units=self.size_of_hidden[1]))
        # model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(activation='relu', input_dim=self.size_of_hidden[1], units=self.size_of_hidden[2]))
        model.add(keras.layers.Dense(units=self.action_count, activation='linear'))

        optimizer = keras.optimizers.Adam(lr=0.002)
        model.compile(loss='mse', optimizer=optimizer)
        return model


    def train(self, X, Y):
        for i in range(len(X)):
            x = np.array([X[i]])
            y = np.array([Y[i]])
            self.model.fit(x, y, batch_size=256, verbose=0)
            # self.async_model.fit(x, y, batch_size=64, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def predict_one(self, state):
        state = np.array(([state]))
        return self.model.predict(state)

    def update_target_network(self):
        self.target_network.set_weights(self.model.get_weights())

    def update_model(self):
        self.model.set_weights(self.async_model.get_weights())
