import os
import numpy as np
import json
import torch
import librosa
from torch.utils.data.dataset import Dataset
from utils import *
import pickle

class TANG(Dataset):
    moves = []
    musics = []
    names = []
    series_uid = []
    uid = []
    def __init__(self, seq_len, dataset_location='', normalize = True, add_centers = True):
        self.seq_len = seq_len
        self.normalize = normalize
        
        print('Loading the dataset...')
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                
                with open(file_path, 'rb') as f:
                    data=pickle.load(f)
                    
                #self.moves = data['moves']
                
                moves=data['moves']
                musics = data['musics']
                movesx = []
                movesy = []
                musicsx = []
                musicsy = []
                for idx, move in enumerate(moves):
                    skeletons = np.array(move['skeletons'])
                    music = np.array(musics[idx])
                    if(add_centers):
                        centers = np.array(move['center'])
                        centers = np.expand_dims(centers,1)
                        skeletons = np.add(skeletons, centers)
                    skeletons = skeletons.reshape(skeletons.shape[0], 69)
                    skeletons = np.transpose(skeletons, (1, 0))
                    skeletons = rolling_window(skeletons, self.seq_len)
                    music = rolling_window(music, self.seq_len)
                    skeletons = np.transpose(skeletons, (1, 2, 0))
                    music = np.transpose(music, (1,2,0))
                    for i in range(skeletons.shape[0]):
                        movesx.append(skeletons[i])
                        movesy.append(skeletons[i])
                        musicsx.append(music[i])
                        musicsy.append(music[i])
                        self.uid.append(i)
                        self.series_uid.append(idx)
                        
                if(self.normalize):
                    movesx = torch.from_numpy(np.array(movesx))
                    self.mean = movesx.mean(dim=(0, 1))
                    self.std = movesx.std(dim=(0, 1))
                    self.moves = (movesx - self.mean) / self.std
                    
                else:
                    self.musics = torch.from_numpy(np.array(musicsy))
                    self.moves = torch.from_numpy(np.array(movesy))
                    self.mean = None
                    self.std = None
                self.names=data['names']
        
    def __getitem__(self, index):
        '''
        move = self.moves[index]
        skeletons = np.array(move['skeletons'])
        centers = np.array(move['center'])
        centers = np.expand_dims(centers,1)
        skeletons = np.add(skeletons,centers)
        skeletons = skeletons.reshape(69,skeletons.shape[0])
        skeletons = rolling_window(skeletons, self.seq_len)
        skeletons = torch.from_numpy(skeletons).permute(1,2,0)
        '''
        skeletons = self.moves[index]
        skeletons = skeletons.type(torch.FloatTensor)
        uid = self.uid[index]
        suid = self.series_uid[index]
        music = self.musics[index]
        music = music.type(torch.FloatTensor)
        return skeletons, music, uid, suid
    
        # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.uid)
            