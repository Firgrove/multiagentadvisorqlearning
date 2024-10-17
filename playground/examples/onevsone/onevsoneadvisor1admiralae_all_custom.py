import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd

import pommerman
from pommerman import agents
from ast import literal_eval
from tqdm import tqdm
import csv
import json

sess = tf.Session()

def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))

agent_list = [
    agents.Advisor_all_custom_ae(201, sess),
    agents.Advisor1admiralae(201, sess),
]

sess.run(tf.global_variables_initializer())
env = pommerman.make('OneVsOne-v0', agent_list)
env.seed(1)

def compute_advisor_correct(row):
    state = np.array(row['State'])
    a2 = row['action2']
    best_action = agent_list[0].act3(state, a2)
    for i in range(4):
        if row[f'Advisor{i}_action'] == best_action:
            row[f'Advisor{i}_correct'] = 1
    
    return row

def main():
    print(pommerman.REGISTRY)
    
    
    
    with open('pommermanonevsonecustomallvs1_rewards.csv', 'w+') as myfile:
        myfile.write('{0}|{1}|{2}\n'.format("Episode", "Reward1(MAD)","Reward2(ExpertQ)"))
    with open('pommermanonevsonecustomallvs1_states.csv', 'w+') as myfile:
        myfile.write('{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}\n'.format("State", "action2", "Advisor0_action", "Advisor1_action", "Advisor2_action", "Advisor3_action", "Advisor0_correct", "Advisor1_correct", "Advisor2_correct", "Advisor3_correct"))
    results = pd.DataFrame(columns=["State", "action2", "Advisor0_action", "Advisor1_action", "Advisor2_action", "Advisor3_action", "Advisor0_correct", "Advisor1_correct", "Advisor2_correct", "Advisor3_correct"])

    cumulative_rewards = []
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    for i_episode in range(20000):
        state = env.reset()


        done = False
        actions = env.act(state)    
        while not done:
            state_new, reward, done, info = env.step(actions)
            a2 = agent_list[0].get_action2()
            actions_new = env.act(state_new) 
            actions2_c, _ = agent_list[0].act2(state_new[0], env.action_space)
            actions_c, _ = agent_list[0].act2(state_new[1], env.action_space) 
            actions2_ = agent_list[1].act2(state_new[0], env.action_space)
            actions_ = agent_list[1].act2(state_new[1], env.action_space)
            advisor_actions = agent_list[0].get_advisor_actions(state_new[0])
            agent_list[0].store(state[0], actions[0], actions[1], reward[0], state_new[0], actions2_c, actions_c)
            agent_list[1].store(state[1], actions[1], actions[0], reward[1], state_new[1], actions_, actions2_)
            agent_list[0].set(actions2_c)
            agent_list[1].set(actions2_)
            state = state_new
            actions = actions_new

            # Format state so we can print to csv
            state0 = featurize(state[0]).tolist()

            row = {
                'State': state0,
                'action2': a2,
                "Advisor0_action": advisor_actions[0], 
                "Advisor1_action": advisor_actions[1],
                "Advisor2_action": advisor_actions[2], 
                "Advisor3_action": advisor_actions[3],
                "Advisor0_correct": 0, 
                "Advisor1_correct": 0, 
                "Advisor2_correct": 0, 
                "Advisor3_correct": 0
            }

            new_row = pd.DataFrame([row])
            results = pd.concat([results, new_row], ignore_index=True)

        agent_list[0].learn()
        agent_list[1].learn()
        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]

        if i_episode % 40 == 0:
            results.to_csv('pommermanonevsonecustomallvs1_states.csv', mode='a', index=False, header=False, sep='|')
            results = pd.DataFrame(columns=["State", "action2", "Advisor0_action", "Advisor1_action", "Advisor2_action", "Advisor3_action", "Advisor0_correct", "Advisor1_correct", "Advisor2_correct", "Advisor3_correct"])
    
        
        print('Episode {} finished'.format(i_episode))
        with open('pommermanonevsonecustomallvs1_rewards.csv', 'a') as myfile:
            myfile.write('{0}|{1}|{2}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1]))
    env.close()
    print(results)
    results.to_csv('pommermanonevsonecustomallvs1_states.csv', mode='a', index=False, header=False, sep='|')

    # Compute advisor correct using chunks. Otherwise not enough ram
    with open('pommermanonevsonecustomallvs1_states_final.csv', 'w+') as myfile:
        myfile.write('{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}\n'.format("State", "action2", "Advisor0_action", "Advisor1_action", "Advisor2_action", "Advisor3_action", "Advisor0_correct", "Advisor1_correct", "Advisor2_correct", "Advisor3_correct"))
    replay = pd.read_csv('pommermanonevsonecustomallvs1_states.csv', delimiter='|', chunksize=1000, converters={'State': literal_eval})
    for chunk in tqdm(replay):
        chunk = chunk.apply(compute_advisor_correct, axis=1)
        print(chunk)
        chunk.to_csv('pommermanonevsonecustomallvs1_states_final.csv', mode='a', index=False, header=False, sep='|')


if __name__ == '__main__':
    main()