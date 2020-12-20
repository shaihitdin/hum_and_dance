import copy
import torch
import torch.nn as nn
import time
from tqdm import tqdm

def train_model(model, device, data_loaders, dataset_sizes, criterion, optimizer, num_epochs=25, batch_size = 20):
    losses = {'dancetrain': [], 'danceval': [],'train':[], 'val':[], 'musictrain':[], 'musicval':[]}
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            if phase == 'val':
                # scheduler.step()
                model.eval()
            else:
                model.train()

      
            running_loss = 0.0
            dance_loss = 0.0
            music_loss = 0.0
            with tqdm(total = dataset_sizes[phase]//batch_size + 1) as pb:
                for step, (moves, music, uid, series_uid) in enumerate(data_loaders[phase]):
                    moves = moves.to(device)
                    music = music.to(device)
                    mean = music.mean(dim=(0,1))
                    std = music.std(dim=(0,1))
                    for i in range(16):
                        if std[i] == 0:
                            std[i] = 1 
                    #print(mean,std)
                    music = (music-mean)/std
                    moves = (moves-moves.mean(dim=(0,1)))/moves.std(dim=(0,1))      
                    #print(moves.mean(dim=(0,1)), moves.std(dim=(0,1)), music.mean(dim=(0,1)), music.std(dim=(0,1)))    
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase != 'val'):
                        outputs, preds= model(moves)
                        inv_idx = torch.arange(model.sequence_length - 1, -1, -1).long()
                        rec_loss = criterion(outputs, moves[:, inv_idx, :])
                        pred_loss = criterion(preds, music[:, inv_idx, :])
                        loss = rec_loss + pred_loss
                     
                        #if train_predictor:
                        #    encoding = model.encoding.detach()
                        #    encoding = encoding.to(device)
                        #    preds = predictor(encoding)
                        #    pred_loss = criterion(preds, music[:, inv_idx, :])
                       
                        if phase != 'val':
                            loss.backward()
                            optimizer.step()
                            #if train_predictor:
                            #     pred_loss.backward()
                            #     optimizer2.step()
    
                    running_loss += loss.item() 
                    dance_loss += rec_loss.item()
                    music_loss += pred_loss.item()
                    #if train_predictor:
                    #    music_loss += pred_loss.item()
                    pb.update(1)
                  
                    pb.set_description('Epoch {} / {}  {:<5} Loss: {:.4f} rec_loss: {:.4f} pred_loss: {:.4f}'.format(epoch + 1, num_epochs, phase, running_loss/(step+1), dance_loss/(step+1), music_loss/(step+1)))
                   
            
            epoch_loss = running_loss * batch_size / dataset_sizes[phase]
            music_loss = music_loss * batch_size / dataset_sizes[phase]
            dance_loss = dance_loss * batch_size / dataset_sizes[phase]
            losses[phase].append(epoch_loss)
            losses['dance'+phase].append(dance_loss)
            losses['music'+phase].append(music_loss)
            #print('Epoch {} / {}'.format(epoch + 1, num_epochs), '{} Loss: {:4f}'.format(phase, epoch_loss)) 
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model, losses