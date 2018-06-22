import random as rdm

from CriptoQLearning.Action import Action


class Environment:

    def __init__(self):
        pass

    def execute(self, action, balance):
        if action == Action.BUY:
            print("BUY")
            reward = balance - 1
        elif action == Action.SELL:
            print("SELL")
            reward = balance + 1
        else:
            print("CAMATE POFAVO, HOLD")
            reward = balance

        # Reward
        if balance <= 0:
            print("No hay mas dinero")
            isDone = True
            s_ = 'terminal'  # estado terminal
        else:
            isDone = False
            s_ = 'continue'

        return s_, reward, isDone

    def getReward(self):
        return rdm.randint(-2, 2)
