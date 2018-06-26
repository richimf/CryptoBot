import random as rdm

from CriptoQLearning.Action import Action


class Environment:

    def __init__(self, balance, price):
        self.balance = balance
        self.price = price

    def __init__(self, balance, price, reward):
        self.balance = balance
        self.price = price
        self.reward = reward

    @property
    def reward(self):
        return self.reward

    def execute(self, action):
        if action == Action.BUY:
            # print("BUY")
            self.balance = self.balance - self.price
        elif action == Action.SELL:
            # print("SELL")
            self.balance = self.balance + self.reward
        else:
            # print("CAMATE POFAVO, HOLD")
            reward = self.reward

        # Balance conditions
        if self.balance <= 0:
            self.balance = 0
            # print("No hay mas dinero")
            isDone = True
            s_ = 'terminal'  # estado terminal
        else:
            isDone = False
            s_ = 'continue'

        return s_, reward, isDone, self.balance

    def getReward(self):
        return rdm.randint(-2, 2)
