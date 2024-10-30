import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd

import torch
from advisor_selection_model import BinaryClassification

import pommerman
from pommerman import agents
from tqdm import tqdm
import csv
import json

sess = tf.Session()

def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def simplify_state(state):
    board = state["board"].reshape(-1).astype(np.float32)
    position = make_np_float(state["position"])

    enemies = []
    for enemy in state['enemies']:
        coords = np.argwhere(state['board'] == enemy.value).tolist()
        for coord in coords:
            enemies.append(coord)
    if len(enemies) < 2:
        enemies = enemies + [[-1, -1]]*(3 - len(enemies))
    enemies = np.array(enemies).reshape(-1)

    bombs = []
    bomb_list = np.argwhere(state["bomb_life"] != 0).tolist()
    for bomb_coords in bomb_list:
        bomb = bomb_coords
        bomb += [state["bomb_life"][bomb_coords[0], bomb_coords[1]]]
        bombs.append(bomb)
    
    while len(bombs) < 5:
        bombs += [[-1, -1, -1]]
    bombs = np.array(bombs).reshape(-1)

    new_state = np.concatenate((board, position, enemies, bombs))
    if new_state.shape[0] != 87:
        raise ValueError()
    return new_state

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

episodes = 50000

agent_list = [
    agents.MADdm(201, sess, episodes-10000),
    agents.DQNAgent(201, sess)
]

sess.run(tf.global_variables_initializer())
env = pommerman.make('OneVsOne-v0', agent_list)
env.seed(1)

def main():
    print(pommerman.REGISTRY)
    
    
    
    with open('MADdm_vs_DQN_train.csv', 'w+') as myfile:
        myfile.write('{0}|{1}|{2}|{3}|{4}\n'.format("Episode", "Reward1(MAD_DM)","Reward2(DQN)","epsilon", "turns"))

    cumulative_rewards = []
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    turns_to_complete = []
    for i_episode in range(episodes):
        state = env.reset()

        turns = 0
        done = False
        actions = env.act(state)    
        while not done:
            turns += 1
            state_new, reward, done, info = env.step(actions)
            actions_new = env.act(state_new) 
            agent_list[0].store(state[0], actions[0], actions[1], actions_new[1], reward[0], state_new[0], actions_new[0])
            agent_list[1].store(state[1], actions[1], reward[1], state_new[1])
            agent_list[0].set(actions_new[1])
            state = state_new
            actions = actions_new

        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]

        if i_episode < episodes-10000: 
            agent_list[0].learn()
            agent_list[1].learn()

            with open('MADdm_vs_DQN_train.csv', 'a') as myfile:
                myfile.write('{0}|{1}|{2}|{3}|{4}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1], agent_list[0].get_eps(), turns))

        else:
            if i_episode == episodes-10000:
                agent_list[0].executionenv() 
                agent_list[1].executionenv()

                with open('MADdm_vs_DQN_test.csv', 'w+') as myfile:
                    myfile.write('{0}|{1}|{2}|{3}\n'.format("Episode", "Reward1(MAD_DM)","Reward2(DQN)", turns))
                cumulative_rewards[0] = 0
                cumulative_rewards[1] = 0

            with open('MADdm_vs_DQN_test.csv', 'a') as myfile:
                myfile.write('{0}|{1}|{2}|{3}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1]), turns)
        
        
        print('Episode {} finished'.format(i_episode))
        
    env.close()




if __name__ == '__main__':
    main()
