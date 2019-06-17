import numpy as np
import pandas as pd
import os


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.actions_str = [str(i) for i in actions]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        if os.path.exists("qtable.csv"):
            self.q_table = pd.read_csv("qtable.csv", index_col=0)
        else:
            self.q_table = pd.DataFrame(columns=self.actions_str, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return int(action)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        try:
            q_predict = self.q_table.loc[s, str(a)]
        except Exception as e:
            print(e)
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        self.q_table.loc[s, str(a)] += self.lr * (q_target - q_predict)  # update
        # print(self.q_table)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


if __name__ == '__main__':
    if os.path.exists("qtable.csv"):
        df = pd.read_csv("qtable.csv", index_col=0)
        print(df.head(10))
        print(df.loc['[1]', "1"])
    else:
        print(False)
