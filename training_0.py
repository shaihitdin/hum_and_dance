import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data_loader import TANG
from model import AutoEncoderRNN
from train_model import train_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sequence_length = 10

dataset=TANG(seq_len=sequence_length, dataset_location='data/', normalize=True)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.15 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
data_loaders={}
dataset_sizes = {}
data_loaders['train'] = DataLoader(dataset, batch_size=20, sampler=train_sampler)
data_loaders['val'] = DataLoader(dataset, batch_size=20, sampler=test_sampler)
dataset_sizes['val'] = len(test_indices)
dataset_sizes['train'] = len(train_indices)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

log=[]
lrs=[]
input_size = 69
num_epochs = 25

for i in range(-30, 20, 5):
    hidden_size = 16
    num_layers = 2
    learning_rate = pow(10,i/10)
    lrs.append(learning_rate)
    
    model = AutoEncoderRNN(input_size, hidden_size, num_layers, seq_len = sequence_length)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model, losses = train_model(model, device, data_loaders, dataset_sizes, criterion, optimizer, num_epochs=num_epochs)
    log.append(losses)
    del model

print('Done. Saving into pickle...')      
data={}
data['hidden_size'] = hidden_size
data['num_layers'] = num_layers
data['learning_rate'] = lrs
data['log'] = log
name = str(hidden_size)
with open('trained_models/'+name+'.pickle', wb) as f:
    pickle.dump(data, f)