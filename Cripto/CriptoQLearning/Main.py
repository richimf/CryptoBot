# 1 episode =  96 steps
# 1 state is a step every 15 min
# number_bitcoins * [p(t)/p(t-1)-1], price

from CriptoQLearning.Qlearning import QLearning
from CriptoQLearning.Action import Action
from CriptoQLearning.Functions import *

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


# trading
def trade():
    reward = 0
    size_episode = 7  # ventanas cortas de tiempo
    bitcoins = 1000  # mi saldo

    data = getStockDataVec("bitcoin")
    data_length = len(data) - 1

    num_episodes = int(data_length / size_episode)
    inventory = [data[0]]

    # loop for each episode, each episode has 96 states
    for episode in range(num_episodes):
        # initialize s, initial observation, same prob, probability of [buy sell hold]
        s = []
        print(">------ Episode = ", episode)
        # loop for each step in episode
        for t in range(1, size_episode - 1):

            s_ = getState(data, t, size_episode)  # pedazo de data de tamano size_episode

            # RL choose action based on observation
            action = RL.chooseAction(s_)
            # print("Action = ", action)

            # For last episode SELL
            if episode == num_episodes:
                action = Action.SELL

            # If bitcoins is less than 0
            if bitcoins <= 0:
                action = Action.SELL

            # For every state in episode
            if action == Action.BUY:
                if bitcoins / int((data[t])) > 1:
                    bitcoins = bitcoins - data[t]
                    # agregamos a la lista el precio
                    inventory.append(data[t])
            elif action == Action.SELL:
                if len(inventory) > 0 and t < len(inventory):
                    last_value = inventory[len(inventory) - 1]
                    bitcoins = bitcoins + last_value
                    inventory.pop(t)

            done = True if t == data_length - 1 else False

            # maximizar
            reward = reward + bitcoins * ((data[t] / data[t - 1]) - 1)

            # RL learn from this transition
            # RL.learn()
            if len(s) == 5:
                RL.q_table.append((s, action, reward, s_, done))

            # swap observation
            s = s_

            print("Reward = ", formatPrice(reward))
            # print("Balance =", balance)

            if len(RL.q_table) > size_episode:
                RL.learn(size_episode)


if __name__ == "__main__":
    actions = [Action.BUY, Action.SELL, Action.HOLD]
    RL = QLearning(actions)
    trade()
