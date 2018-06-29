import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class QLearning:

    def __init__(self, actions, action_size=3, state_size=6, model_name="", e_greedy=1, is_eval=False):
        self.is_eval = is_eval
        self.actions = actions  # a list of actions [Action.BUY, Action.SELL, Action.HOLD]
        self.action_size = action_size
        self.state_size = state_size

        # Model RN
        self.model_name = model_name
        self.model = load_model("models/" + model_name) if is_eval else self._model()

        # Q-Table memory
        self.q_table = deque(maxlen=1000)

        # Constants
        self.gamma = 0.95
        self.epsilon = e_greedy
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    # The Neural Network
    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def chooseAction(self, state):
        # self.appendNotExistingState(state)
        action = self.action(state)
        action_str = "HOLD"
        if action == 1:
            action_str = "BUY"
        elif action == 2:
            action_str = "SELL"

        print("Action = ", action_str)
        return action

    def action(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    # state, action, reward, next state
    def learn(self, batch_size):
        mini_batch = []
        l = len(self.q_table)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.q_table[i])

        for state, action, reward, next_state, done in mini_batch:
            print(state, action, reward, next_state, done)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # print(state[0])
            # state = pd.DataFrame(np.transpose(state[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


"""
    def appendNotExistingState(self, state):
        if state not in self.q_table.index:
            # append new state to q-table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
"""
