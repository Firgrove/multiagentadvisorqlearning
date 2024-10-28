from flask import Flask, request

import d3rlpy
from d3rlpy.algos import DQNConfig, DiscreteDecisionTransformerConfig

import json
import numpy as np
import pandas as pd

from ast import literal_eval

# Start server
app = Flask(__name__)

# Initialise model
model = DQNConfig().create(device="cuda:0")

def __init__():
    # Load dataset
    data = pd.read_csv('pommermanonevsonecustomallvs1_states.csv', sep='|')

    # Convert states to np arrays
    data['State_small'] = data['State_small'].apply(literal_eval)
    data['State_small'] = data['State_small'].apply(np.array)
    # data['State'] = data['State'].apply(literal_eval)
    # data['State'] = data['State'].apply(np.array)
    # print(data['State'].iloc[0].shape)

    # Remove states where no advisor was used
    data = data[data['advisor_used'] != -1]

    # Now create the dataset
    observations = np.stack(data['State_small'].values)
    actions = np.stack(data['advisor_used'].values)
    rewards = np.stack(data['reward'].values)
    terminals = np.stack(data['done'].apply(int))

    print(observations.shape, actions.shape, rewards.shape, terminals.shape)

    # Create dataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    
    # start training
    # epchs is n_steps/n_steps_per_epoch
    model.fit(
        dataset,
        n_steps=15000,
        n_steps_per_epoch=1000,
    )

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    print(f'received request: {request.json}')  # TODO: Remove for performance
    obs = np.array(json.loads(request.json)['obs'])
    obs = np.expand_dims(obs, axis=0)
    return json.dumps({'output': model.predict(obs).item()})

if __name__ == "__main__":
    __init__()
    app.run()