import torch
import torch.nn as nn
import numpy as np

device=torch.device('cuda:0')

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first = True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.relu = nn.ReLU()

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        return out[:, -1, :].unsqueeze(1)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional, batch_first = True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=batch_first,
                            dropout=0.2, bidirectional=bidirectional)

        # initialize weights
        #nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        #nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        
    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len = 10, bidirectional=False, batch_size = 20, batch_first = True):
        super(AutoEncoderRNN, self).__init__()
        self.sequence_length = seq_len
        self.encoding = torch.zeros((batch_size, seq_len, hidden_size))
        self.dense1 = nn.Linear(input_size, hidden_size*2) 
        self.encoder = EncoderRNN(hidden_size*2, hidden_size, num_layers, bidirectional, batch_first = batch_first)
        self.decoder = DecoderRNN(hidden_size, hidden_size*2, num_layers, bidirectional, batch_first = batch_first)
        self.dense2 = nn.Linear(hidden_size*2, input_size)
        
        self.predictor = DecoderRNN(hidden_size, output_size, num_layers, bidirectional, batch_first = batch_first)
        self.dense3 = nn.Linear(output_size, output_size)
    
        
    def forward(self, x):
        x = self.dense1(x)
        encoded_x = self.encoder(x).expand(-1, self.sequence_length, -1)
        self.encoding = encoded_x
        decoded_x = self.decoder(encoded_x)
        y = self.predictor(encoded_x)
        predicted_y = self.dense3(y)
        decoded_x = self.dense2(decoded_x)
        return decoded_x, predicted_y