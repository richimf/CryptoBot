# 1 episode =  96 steps
# 1 state is a step every 15 min
# number_bitcoins * [p(t)/p(t-1)-1], price

from CriptoQLearning.Functions import *
from CriptoQLearning.Action import Action


class Trade:

    def __init__(self, data, size_episode=7, coins=0, balance=0, q_learning=None, percentage=0.25):
        self.data = data
        self.balance = balance
        self.coins = coins
        self.percentage = percentage  # percentage for investment
        self.size_episode = size_episode
        self.RL = q_learning

    def trade(self):
        reward = 0
        data_length = len(self.data) - 1

        num_episodes = int(data_length / self.size_episode)
        inventory = [self.data[0]]

        # plot data
        # plot(data)

        # loop for each episode
        for episode in range(num_episodes):
            # initialize s, initial observation, same prob, probability of [buy sell hold]
            s = []
            print(">------ Episode = ", episode)
            # loop for each step in episode
            for t in range(1, self.size_episode - 1):

                s_ = getState(self.data, t, self.size_episode)  # pedazo de data de tamano size_episode

                # RL choose action based on observation
                action = self.RL.chooseAction(s_)
                # print("Action = ", action)

                # For last episode SELL
                if episode == num_episodes:
                    action = Action.SELL

                # If bitcoins is less than 0
                if self.coins <= 0:
                    action = Action.SELL

                # For every state in episode
                if action == Action.BUY:
                    # update number of coins and balance
                    coin_price = self.data[t]
                    investment = self.balance * self.percentage
                    purchased_coins = investment / coin_price
                    self.coins = self.coins + purchased_coins  # update coins
                    self.balance = self.balance - investment  # update balance

                    if self.coins / int((self.data[t])) > 1:
                        self.coins = self.coins - self.data[t]
                        # agregamos a la lista el precio
                        inventory.append(self.data[t])
                elif action == Action.SELL:
                    if len(inventory) > 0 and t < len(inventory):
                        last_value = inventory[len(inventory) - 1]
                        self.coins = self.coins + last_value
                        inventory.pop(t)

                done = True if t == data_length - 1 else False

                # maximizar
                reward = reward + self.coins * ((self.data[t] / self.data[t - 1]) - 1)
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
