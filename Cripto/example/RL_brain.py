"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

class QLearningTable:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :] # get action fron q_table via observation
            print("\n>---- Choose action....")
            print("> Q table = ")
            print(self.q_table)
            print("> Q table with observation = ", observation)
            print("> state_action = ", state_action)
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax() # el id de la accion con mayor valor
            print("> action if = ", action)
        else:
            # choose random action
            action = np.random.choice(self.actions)
            print("> action else = ", action)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # q_target = r + gamma*max_a[Q(S',a)]
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        # Q(S,A) <- Q(S,A) + alfa[r + gamma*max_aQ(S',a) - Q(S,A)]
        # Q(S,A) <- Q(S,A) + alfa[q_target - Q(S,A)]  # alfa is the learning rate: "self.lr"
        # Q(S,A) <- Q(S,A) + self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
