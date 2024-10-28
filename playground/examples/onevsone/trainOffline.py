import numpy as np
import pandas as pd

import d3rlpy
from d3rlpy.algos import DQNConfig, DiscreteDecisionTransformerConfig

from ast import literal_eval

def run():
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
    terminals = np.stack(data['reward'].apply(abs))

    print(observations.shape, actions.shape, rewards.shape, terminals.shape)

    # Create dataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    # Initialise model
    model = DQNConfig().create(device="cuda:0")
    # start training
    # epchs is n_steps/n_steps_per_epoch
    model.fit(
        dataset,
        n_steps=15000,
        n_steps_per_epoch=1000,
    )

    model.save_model('AS_model.pt')

    for i in range(15):
        print(model.predict(np.expand_dims(observations[i], axis=0)))

if __name__ == "__main__":
    run()