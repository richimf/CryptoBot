# 1 episode =  96 steps
# 1 state is a step every 15 min

import random as rdm

from CriptoQLearning.Qlearning import QLearning
from CriptoQLearning.Environment import Environment
from CriptoQLearning.Action import Action
from CriptoQLearning.Episode import Episode

# TODO: Alimentar con RN
# TODO: Prices of Bitcoin, se puede borrar
data = []
for i in range(100):
    data.append(rdm.randint(5000, 290000))
print(data)


def trade():
    balance = 2000  # mi saldo
    episodes = []

    # creating episodes of 96 steps
    dummy_step = [0, 0, 0]

    for _ in range(10):
        steps = []
        for _ in range(96):
            steps.append(dummy_step)
        episodes = Episode(steps)

    print(episodes)
    num_episodes = episodes.count()  # 1 episode =  96 steps , step = [0.3, 0.2, 0.5]

    # loop for each episode
    for episode in range(num_episodes):     # TODO, AQUI DEBE HABER UN ARRAY DE EPISODES, CADA EPISODE CON 96 STEPS
        # initialize s, initial observation, same prob, probability of [buy sell hold]
        s = [0.25, 0.25, 0.5]
        # loop for each step in episode
        for step in range(1, 96):
            # RL choose action based on observation
            action = RL.choose_action(str(s))
            print("\nObservation S = ", str(s))
            print("Action = ", action)

            # RL take action and get next observation and reward
            s_, reward, done = Env.execute(action, balance)
            print("Reward =", reward)
            print("Observation s_ =", s_)

            balance = reward
            print("Balance = ", balance)

            # RL learn from this transition
            RL.learn(str(s), action, reward, str(s_))

            # swap observation
            s = s_


if __name__ == "__main__":
    actions = [Action.BUY, Action.SELL, Action.HOLD]
    Env = Environment()
    RL = QLearning(actions)
    trade()
