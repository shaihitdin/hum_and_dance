import os
import json
import librosa
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data_loader import TANG
from utils import *
from model import AutoEncoderRNN, DecoderRNN
from train_model import train_model

device=torch.device('cuda:0')
sequence_length = 50
batch_size = 100

dataset=TANG(seq_len=sequence_length, dataset_location='data/', normalize=False)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.15 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
data_loaders={}
dataset_sizes = {}
data_loaders['train'] = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
data_loaders['val'] = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
dataset_sizes['val'] = len(test_indices)
dataset_sizes['train'] = len(train_indices)
print("Number of training/test patches:", (len(train_indices),len(test_indices)), dataset.__len__())

num_epochs = 100
learning_rate = 0.001
input_size = 69
output_size = 16
hidden_size = 8
#model = nn.LSTM(input_size, output_size, num_layers = 3, bidirectional = False, batch_first = True)
model = AutoEncoderRNN(input_size, hidden_size, output_size, num_layers = 2, seq_len = sequence_length, batch_size = batch_size, batch_first=True)
#predictor = DecoderRNN(hidden_size, output_size, num_layers = 2, bidirectional =False, batch_first=True)
model = model.to(device)
#predictor = predictor.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#model, losses = train_model(model, predictor, device, data_loaders, dataset_sizes, criterion, optimizer1, optimizer2, num_epochs=num_epochs, batch_size = batch_size, train_predictor = False)
model, losses = train_model(model, device, data_loaders, dataset_sizes, criterion, optimizer, num_epochs=num_epochs, batch_size = batch_size)
with open('trained_models/model12_2-35AM.pickle', 'wb') as f:
    pickle.dump(model, f)
with open('trained_models/loss12_2-35AM.pickle', 'wb') as f:
    pickle.dump(losses, f)