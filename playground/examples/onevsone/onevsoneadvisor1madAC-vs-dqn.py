import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd

import pommerman
from pommerman import agents
import csv
import json

sess = tf.Session()



agent_list = [
    agents.Advisor_all_custom_ae(201, sess),
    agents.DQNAgent(201, sess),
]

sess.run(tf.global_variables_initializer())
env = pommerman.make('OneVsOne-v0', agent_list)
env.seed(1)

def compute_advisor_correct(row):
    state = row['State']
    best_action = agent_list[0].act(state, env.action_space, no_advisor=True)
    if best_action == row['Advisor_action']:
        return 1
    
    return 0

def main():
    print(pommerman.REGISTRY)
    
    
    
    with open('pommermanonevsonecustomallvs1_rewards.csv', 'w+') as myfile:
        myfile.write('{0}|{1}|{2}\n'.format("Episode", "Reward1(Custom)","Reward2(ExpertQ)"))
    with open('pommermanonevsonecustomallvs1_states.csv', 'w+') as myfile:
        myfile.write('{0}|{1}|{2}|{3}\n'.format("State", "Advisor_action", "Advisor_used", "Advisor_correct"))
    results = pd.DataFrame(columns=["State", "Advisor_action", "Advisor_used", "Advisor_correct"])

    cumulative_rewards = []
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    for i_episode in range(10000):
        state = env.reset()


        done = False
        actions = env.act(state)    
        while not done:
            state_new, reward, done, info = env.step(actions)
            actions_new = env.act(state_new) 
            actions2_c, advisor_used = agent_list[0].act2(state_new[0], env.action_space)
            actions_c, _ = agent_list[0].act2(state_new[1], env.action_space) 
            actions2_ = agent_list[1].act2(state_new[0], env.action_space)
            actions_ = agent_list[1].act2(state_new[1], env.action_space)
            agent_list[0].store(state[0], actions[0], actions[1], reward[0], state_new[0], actions2_c, actions_c)
            agent_list[1].store(state[1], actions[1], reward[1], state_new[1])
            agent_list[0].set(actions2_c)
            agent_list[1].set(actions2_)
            state = state_new
            actions = actions_new

            # Format state so we can print to csv
            state0 = {}
            for key, val in state[0].items():
                state0[key] = val
                if isinstance(val, np.ndarray):
                    state0[key] = val.tolist()
                if isinstance(val, tuple):
                    state0[key] = [int(i) for i in val]
                if key == 'teammate':
                    state0[key] = val.value
                if key =='enemies':
                    state0[key] = [enemy.value for enemy in val]

            results_dict = {
                "State": json.dumps(state0),
                "Advisor_action": actions2_c,
                "Advisor_used": advisor_used,
                "Advisor_correct": 0
            }
            new_row = pd.DataFrame([results_dict])
            results = pd.concat([results, new_row])

        agent_list[0].learn()
        agent_list[1].learn()
        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]

        if i_episode % 40 == 0:
            results.to_csv('pommermanonevsonecustomallvs1_states.csv', mode='a', index=False, header=False, sep='|')
            results = pd.DataFrame(columns=["State", "Advisor_action", "Advisor_used", "Advisor_correct"])
    
        
        print('Episode {} finished'.format(i_episode))
        with open('pommermanonevsoneexpert1offpolicy_rewards.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1]))
    env.close()
    results.to_csv('pommermanonevsonecustomallvs1_rewards.csv', mode='a', index=False, header=False, sep='|')

    replay = pd.read_csv('pommermanonevsonecustomallvs1_states.csv', delimiter='|')
    print(replay.head())
    replay['State'] = replay['State'].apply(json.loads)
    replay['Advisor_correct'] = replay.apply(compute_advisor_correct, axis=1)
    print(replay.head())
    #replay['State'] = replay['State'].apply(json.dumps)
    replay.to_csv('pommermanonevsonecustomallvs1.csv', sep='|')


if __name__ == '__main__':
    main()
