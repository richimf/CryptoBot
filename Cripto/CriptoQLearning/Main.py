import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from CriptoQLearning.Trade import Trade
from CriptoQLearning.Action import Action
from CriptoQLearning.Functions import *
from CriptoQLearning.Qlearning import QLearning as RL


def plot(data, title='Crypto currency chart', ylabel='Price (BTC-USD)'):
    # Data for plotting
    t = range(0, len(data))
    s = data

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure() and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.plot(t, s)

    # plot chart
    ax.set(xlabel='time (days)', ylabel=ylabel, title=title)
    ax.grid()
    fig.savefig("data.png")
    plt.show()


if __name__ == "__main__":
    # Inputs
    balance = 200
    num_bitcoins = 3
    size_episode = 7

    # Data
    data = getStockDataVec("bitcoin")  # CSV data
    if len(data) == 0:
        print("No data")

    # Show data
    # plot(data)

    # Setup Reinforcement Learning
    actions = [Action.SELL, Action.BUY, Action.HOLD]
    brain = RL(actions)

    # Trading
    t = Trade(data, size_episode, num_bitcoins, balance, brain)
    t.trade()
