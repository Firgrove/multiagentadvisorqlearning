from maze_env import Maze
from RL_brain_admiraldm_advisor4 import QLearningTable
import csv

with open('advisor4.csv', 'w+') as myfile:
    myfile.write('{0},{1},{2}\n'.format("Episode", "Reward-Agent1","Reward2-Agent2"))

def update():
    cumulative_reward = 0
    for episode in range(2000):
        observation, observation2 = env.reset()
        action_tmp = 0
        action2_tmp = 0
        action = 0
        action2 = 0
        while True:
            env.render()



            

            
            observation_, observation2_, reward, reward2, done = env.step(action, action2)
            
            
            if observation_ != 'terminal': 
                observation_copy = observation_.copy()
                observation_copy2 = observation_.copy()
            else: 
                observation_copy = observation_
                observation_copy2 = observation_

            if observation2_ != 'terminal': 
                observation2_copy = observation2_.copy()
                observation2_copy2 = observation2_.copy()
            else: 
                observation2_copy = observation2_
                observation2_copy2 = observation2_

            action_greedy, action2_greedy = RL.choose_greedy_action(observation_copy, observation2_copy, action_tmp, action2_tmp)

            action_, action2_ = RL.choose_action(observation_copy2, observation2_copy2, action_greedy, action2_greedy, episode)


            observationcopy = observation.copy()
            observation2copy = observation2.copy()
            
            if observation_ != 'terminal': 
                observation_copy = observation_.copy()
            else: 
                observation_copy = observation_

            if observation2_ != 'terminal': 
                observation2_copy = observation2_.copy()
            else: 
                observation2_copy = observation2_
            
            RL.learn(observationcopy, observation2copy, action, action2, reward, reward2, observation_copy, observation2_copy, action_, action2_)

            action = action_
            action2 = action2_
            action_tmp = action
            action2_tmp = action2
            observation = observation_
            observation2 = observation2_
            
            if done:
                break
        cumulative_reward = cumulative_reward + reward
        with open('advisor4.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(episode, reward, reward2))
        print('Episode {} finished'.format(episode))
    
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)), actions2 = list(range(env.n_actions)))
    env.after(2000, update)
    env.mainloop()
