from tqdm import tqdm
import librosa
import numpy as np
import os
import json

moves = []
musics = []
configs = []
series_uid = []
print('Loading the dataset...')

with tqdm(total=61) as pbar:
    for idx, (root, dirs, files) in enumerate(os.walk('/data01/tima/tangtao', topdown=False)):
        i=0
        for name in dirs:
            move_file=os.path.join(root,name,"skeletons.json")
            with open(move_file) as json_file:
                move_data = json.load(json_file)

            config_file=os.path.join(root,name,"config.json")
            with open(config_file) as json_file:
                config_data = json.load(json_file)

            #torchaudio supports loading sound files in the mp3 format. Waveform is the resulting raw audio signal.

            y, sr= librosa.load(os.path.join(root,name,"audio.wav"),sr=12785)
            mfcc = librosa.feature.mfcc(y, sr, n_mfcc=3)
            mfcc_delta = librosa.feature.delta(mfcc)
            chroma_q = librosa.feature.chroma_cqt(y, sr, n_chroma=4)
            onset_strength= librosa.onset.onset_strength(y,sr)
            tempogram = librosa.feature.tempogram(y, sr, onset_envelope=onset_strength, win_length=5)
            music_features=np.concatenate((mfcc, mfcc_delta, chroma_q, tempogram,onset_strength.reshape((1,mfcc.shape[1]))), axis=0)
            
            series_uid.append(i)
            musics.append(music_features)
            moves.append(move_data)
            configs.append(config_data)

            assert (len(musics) == len(moves) == len(configs) == len(series_uid))
            del move_data
            del y
            del config_data
            pbar.update(1)
            i+=1
print('Done. Saving into pickle...')      
data={}
data['moves']=moves
data['musics']=musics
data['configs']=configs
data['series_uid']=series_uid
with open('data/tang.pickle','wb') as f:
    pickle.dump(data, f)
    
'''
series_uid=[]
names = []
moves = []
music = []
print('Loading the dataset...')

with tqdm(total=61) as pbar:
    for idx, (root, dirs, files) in enumerate(os.walk('/data01/tima/tangtao', topdown=False)):
        i=0
        for name in dirs:
            move_file=os.path.join(root,name,"skeletons.json")
            with open(move_file) as json_file:
                move_data = json.load(json_file)
            config_file=os.path.join(root,name,"config.json")
            with open(config_file) as json_file:
                config_data = json.load(json_file)
            y, sr= librosa.load(os.path.join(root,name,"audio.wav"),sr=12800)
            mfcc = librosa.feature.mfcc(y, sr, n_mfcc=3, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfcc)
            chroma_q = librosa.feature.chroma_cqt(y, sr, n_chroma=4, hop_length=512)
            onset_strength= librosa.onset.onset_strength(y,sr, hop_length=512)
            tempogram = librosa.feature.tempogram(y, sr, onset_envelope=onset_strength, win_length=5, hop_length=512)
            music_features=np.concatenate((mfcc, mfcc_delta, chroma_q, tempogram,onset_strength.reshape((1,mfcc.shape[1]))), axis=0)
            move_len = move_data['length']
            start_idx = config_data['start_position']
            end_idx = config_data['end_position']
            end_idx = min(end_idx-start_idx, move_len)
            move_data['skeletons'] = move_data['skeletons'][0:end_idx]
            move_data['length'] = end_idx
            move_data['center'] = move_data['center'][0:end_idx]
            series_uid.append(i)
            music.append(music_features[:,start_idx:start_idx + end_idx])
            moves.append(move_data)
            names.append(name)
            pbar.update(1)
            i+=1
print('Done. Saving into pickle...')      
data={}
data['moves']=moves
data['musics']=music
data['names']=names
data['series_uid']=series_uid
with open('data/tang.pickle','wb') as f:
    pickle.dump(data, f)
'''