from rl_brain import QLearningTable

def update():
    for episode in range(100):
        # initial observation
        observation = ''

        while True:

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')


def RL_update(q_table, dataset):
    for episode in range(100):
        process = [0]
        state = str(process)
        for i in range(10):
            pass



if __name__ == "__main__":
    q_table = QLearningTable(actions=list(range(n_actions)))
