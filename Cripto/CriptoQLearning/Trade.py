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
        window = 0  # es la ventana de tiempo
        data_length = len(self.data) - 1

        num_episodes = int(data_length / self.size_episode)
        history_investments = [self.data[0]]
        rewards = []

        # plot data
        # plot(data)

        # loop for each episode
        for episode in range(num_episodes):
            # initialize s, initial observation, same prob, probability of [buy sell hold]
            s = [0, 0, 0, 0, False]
            print(">------ Episode = ", episode)
            # loop for each step in episode
            for t in range(1, self.size_episode - 1):

                print("\n ------")
                s_ = getState(self.data, t, self.size_episode)  # pedazo de data de tamano size_episode

                # RL choose action based on observation
                action = self.RL.chooseAction(s_)
                print("Action = ", action)

                # If bitcoins is less than 0
                if self.coins <= 0:
                    action = Action.SELL

                # For every state in episode
                # BUY = 1,  SELL = 2, HOLD = other
                current_coin_price = self.data[t]
                if action == 1:  # BUY
                    # update number of coins and balance
                    investment = self.balance * self.percentage
                    purchased_coins = investment / current_coin_price
                    self.coins += purchased_coins  # update coins
                    self.balance = self.balance - investment  # update balance

                    # add it to investments list
                    history_investments.append(self.data[t])

                elif action == 2:  # SELL
                    # you must have some investment in order to sell
                    if len(history_investments) > 0:
                        last_value = self.data[t]  # history_investments[len(history_investments) - 1]
                        faction_coins = self.coins * self.percentage
                        self.balance += faction_coins * last_value  # update balance
                        self.coins -= faction_coins
                        history_investments.pop(len(history_investments) - 1)

                # For last episode SELL ALL!
                if episode == num_episodes and t == self.size_episode - 1:
                    action = 2  # SELL

                if self.balance <= 0:
                    done = True
                else:
                    done = True if t == data_length - 1 else False

                # Set maximum reward
                if action != 1 and action != 2:  # HOLD
                    reward = reward
                else:
                    reward = self.coins * ((self.data[t] / self.data[t - 1]) - 1)
                    self.balance = self.balance + reward
                    # rewards.append(reward)

                # rewards.append(reward)
                # RL learn from this transition
                # RL.learn()
                if len(s_) > 0:
                    print("s_[0] = ", s_[0])
                    self.RL.q_table.append((s, action, reward, s_[0], done))

                if len(history_investments) > len(self.RL.q_table):
                    self.RL.learn(window, window + self.size_episode)

                window = window + self.size_episode

                if len(self.RL.q_table) > self.size_episode:
                    self.RL.learn(self.size_episode, window)

                # swap state
                s = s_

                print("Reward = ", formatPrice(reward))
                print("Balance = ", formatPrice(self.balance))
                rewards.append(self.balance)
                print("Coins = ", self.coins)

                # print("Balance =", balance)

        return rewards
