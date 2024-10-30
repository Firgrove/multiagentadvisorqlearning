import torch
from advisor_selection_model import BinaryClassification

model = BinaryClassification()
print('loading model state')
print(torch.load('selection_model.pth', weights_only=False))
model.load_state_dict(torch.load('selection_model.pth', weights_only=False))