# 1 episode =  96 steps
# 1 state is a step every 15 min
# number_bitcoins * [p(t)/p(t-1)-1], price

from CriptoQLearning.Functions import *
from CriptoQLearning.Action import Action

"""
Arguments:
    arg[1]: is batch (should be 96)
    arg[2]: is number of episodes
    arg[3]: the goal(limit of profit)
    arg[4]: the balance
if len(sys.argv) != 4:
    print("Error, args number do not match")
    exit()

batch, num_episodes, goal, balance = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
"""


class Trade:
    reward = 0

    def __init__(self, data, size_episode=7, bitcoins=0, balance=0, q_learning=None):
        self.data = data
        self.balance = balance
        self.bitcoins = bitcoins
        self.size_episode = size_episode
        self.RL = q_learning

    def trade(self):
        # inputs
        reward = 0
        data = getStockDataVec("bitcoin")  # CSV data
        if len(data) == 0:
            print("Data is empty")
            pass

        data_length = len(data) - 1

        num_episodes = int(data_length / self.size_episode)
        inventory = [data[0]]

        # plot data
        # plot(data)

        # loop for each episode
        for episode in range(num_episodes):
            # initialize s, initial observation, same prob, probability of [buy sell hold]
            s = []
            print(">------ Episode = ", episode)
            # loop for each step in episode
            for t in range(1, self.size_episode - 1):

                s_ = getState(data, t, self.size_episode)  # pedazo de data de tamano size_episode

                # RL choose action based on observation
                action = self.RL.chooseAction(s_)
                # print("Action = ", action)

                # For last episode SELL
                if episode == num_episodes:
                    action = Action.SELL

                # If bitcoins is less than 0
                if self.bitcoins <= 0:
                    action = Action.SELL

                # For every state in episode
                if action == Action.BUY:
                    if self.bitcoins / int((data[t])) > 1:
                        self.bitcoins = self.bitcoins - data[t]
                        # agregamos a la lista el precio
                        inventory.append(data[t])
                elif action == Action.SELL:
                    if len(inventory) > 0 and t < len(inventory):
                        last_value = inventory[len(inventory) - 1]
                        self.bitcoins = self.bitcoins + last_value
                        inventory.pop(t)

                done = True if t == data_length - 1 else False

                # maximizar
                reward = reward + self.bitcoins * ((data[t] / data[t - 1]) - 1)
                # rewards.append(reward)
                # RL learn from this transition
                # RL.learn()
                if len(s) == 5:
                    self.RL.q_table.append((s, action, reward, s_, done))

                # swap observation
                s = s_

                print("Reward = ", formatPrice(reward))
                # print("Balance =", balance)

                if len(self.RL.q_table) > self.size_episode:
                    self.RL.learn(self.size_episode)

        return reward
