import torch
import torch.nn as nn

class basic(nn.Module):
    
    def __init__(self):
        super(basic, self).__init__()

        self.linear1 = nn.Linear(3, 50)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(50, 50)
        self.linear3 = nn.Linear(50, 1)
        #self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

# From https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()        # Number of input features is 12.
        self.layer_1 = nn.Linear(23, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 3) 
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x