import pandas as pd
from advisor_selection_model import basic, BinaryClassification
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from ast import literal_eval

import pommerman
from pommerman import agents

from torch.utils.data import Dataset, DataLoader

import json

# epochs = 200

# sess = tf.Session()
# agent = agents.Advisor_all_custom_ae(201, sess)
# agent.restore_model('agent_checkpoints/ae_done.tf')
# print(agent)

tqdm.pandas()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_accuracy(preds, targets):
    pass

def state_to_np_array(row):
    keys_to_change = ['position', 'board', 'bomb_blast_strength', 'bomb_life', 'bomb_moving_direction', 'flame_life', 'ammo']
    array = np.array([row['Advisor_used']])
    for key in keys_to_change:
        new = np.array(row['State'][key]).ravel()
        array = np.concatenate([array, new])
    
    return array.ravel().tolist() + [row['Advisor_correct']]

class ae_dataset(Dataset):
    def __init__(self, df):
        scaler = StandardScaler()
        # iterator=True, chunksize=1,
        df['State'] = df['State'].progress_apply(literal_eval)
        df['State'] = df['State'].map(np.array)
        df = df.reset_index()
        # self.df['State'] = self.df[['State', 'Advisor_used']].progress_apply(state_to_np_array, axis=1)
        self.dataset = np.empty((df.shape[0], 205))
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            # print(row['State'].shape)
            new_row = np.concatenate([row['State'], np.array([row['Advisor0_correct'], row['Advisor1_correct'], row['Advisor2_correct'], row['Advisor3_correct']])]).ravel()
            self.dataset[i,:] = new_row
        

        # TODO: Normalise states using standard scalar
        
        

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.dataset[idx, :-4]), torch.FloatTensor(self.dataset[idx, -4:])

path='pommermanonevsonecustomallvs1_states_final.csv'
train, test = train_test_split(pd.read_csv(path,  sep='|', index_col=None).sample(frac=0.5), test_size=0.1)
print(train)

epochs = 10
dataset_train = ae_dataset(train)
dataset_test = ae_dataset(test)
train_loader = DataLoader(dataset_train, batch_size=512, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True, drop_last=True)
model = BinaryClassification().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()
loss_list = []
output_list = []

for i in range(epochs):
    for batch, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        output_list.append(outputs)

        #print(outputs)
        loss = criterion(outputs, y_batch)
        # acc = accuracy(outputs, y_batch)
        loss_list.append(loss.item())
        # acc_list.append(acc)
        # means.append(torch.mean(outputs).item())
        # stds.append(torch.std(outputs).item())
        loss.backward()
        optimizer.step()
        print(f"epoch: {i}, batch: {batch}, loss: {loss}")

plt.plot(loss_list)
plt.show()

print(torch.sigmoid(output_list[-1]))

y_test = []
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader):
        y_test = y_test + list(y_batch)
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
accuracy = sum([np.abs(a-b) for a, b in zip(y_test, y_pred_list)]) / len(y_test)
print(f'accuracy: {accuracy}')
print(confusion_matrix(y_test, y_pred_list))  
print(classification_report(y_test, y_pred_list))

