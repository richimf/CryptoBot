import matplotlib.pyplot as plt

from CriptoQLearning.Trade import Trade
from CriptoQLearning.Functions import *
from CriptoQLearning.Qlearning import QLearning as RL


# Data for plotting
def createChart(data, title='Crypto currency chart', xlabel='time (days)', ylabel='Price (BTC-USD)', color='r'):
    t = range(0, len(data))
    fig, ax = plt.subplots()
    ax.plot(t, data, color)

    # plot chart
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    # fig.savefig("data.png")
    plt.show()
    # return ax


if __name__ == "__main__":
    # Inputs
    balance = 200
    num_bitcoins = 6.233412
    size_episode = 32

    # Data
    data = getStockDataVec("bitcoin")  # CSV data
    if len(data) == 0:
        print("No data")

    # Setup Reinforcement Learning
    brain = RL()

    # Trading
    t = Trade(data, size_episode, num_bitcoins, balance, brain)
    history = t.trade()

    # Plot data
    createChart(data)
    createChart(history, 'Reward chart', 'time', 'Balance', 'b')
