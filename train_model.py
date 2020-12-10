import copy
import torch
import torch.nn as nn
import time
from tqdm import tqdm

def train_model(model, device, data_loaders, dataset_sizes, criterion, optimizer, num_epochs=25, batch_size = 20):
    losses = {'train': [], 'val': []}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    with tqdm(total = num_epochs) as pb:
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'val':
                    # scheduler.step()
                    model.eval()
                else:
                    model.train()
    
                running_loss = 0.0
                
                for step, (moves, uid, series_uid) in enumerate(data_loaders[phase]):
                    moves = moves.to(device)
    
                    optimizer.zero_grad()
    
                    with torch.set_grad_enabled(phase != 'val'):
                        outputs = model(moves)
                        inv_idx = torch.arange(model.sequence_length - 1, -1, -1).long()
                        loss = criterion(outputs, moves[:, inv_idx, :])
        
                        if phase != 'val':
                            loss.backward()
                            optimizer.step()
    
                    running_loss += loss.item() 
    
                epoch_loss = running_loss * batch_size / dataset_sizes[phase]
                
                losses[phase].append(epoch_loss)
    
                print('Epoch {} / {}'.format(epoch + 1, num_epochs), '{} Loss: {:4f}'.format(phase, epoch_loss)) 
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            pb.update(1)
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model, losses