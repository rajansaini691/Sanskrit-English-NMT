import torch
import numpy as np
import random
from config import Config
from model import Transformer
from torch import nn
from torch import optim

class Data(torch.utils.data.Dataset):
  def __init__(self):

    train_x_data = []
    for i in range(80): #samples
      train_x_data.append([random.randint(2, 42) for _ in range(10)])#sequence length
 
    for x in train_x_data:
        random.shuffle(x)
    
    train_data = [(np.array([x]),np.array([x[::-1]])) for x in train_x_data]

    self.x = None
    self.y = None
    for x,y in train_data:
      if type(self.x) == type(None) and type(self.y) == type(None):
        self.x = x.T 
        self.y = y.T
      else:
        self.x = np.concatenate((self.x, x.T), axis=1)
        self.y = np.concatenate((self.y, y.T), axis=1)

    self.x = [torch.as_tensor(arr) for arr in self.x]
    self.y = [torch.as_tensor(arr) for arr in self.y]
    self.num_samples = len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

  def __len__(self):
    return self.num_samples

def train(model, optimizer, criterion, iterator):
    
    model.train()
    
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, trg = batch
        src, trg = src.to(Config.DEVICE), trg.to(Config.DEVICE)
        
        optimizer.zero_grad()
        output = model(src, trg[:-1,:])

        loss = criterion(output.transpose(0, 1).transpose(1, 2), trg[1:,:].transpose(0, 1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, criterion, iterator):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():    
        for i, batch in enumerate(iterator):
            src, trg = batch
            src, trg = src.to(Config.DEVICE), trg.to(Config.DEVICE)

            
            output = model(src, trg[:-1,:])
            loss = criterion(output.transpose(0, 1).transpose(1, 2), trg[1:,:].transpose(0, 1))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def run_sanity_check():
    pin_memory = True
    num_workers = 2

    train_data = Data()
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=num_workers, shuffle=True,
                            batch_size=Config.BATCH_SIZE)

    val_data = Data()
    val_loader = torch.utils.data.DataLoader(val_data, num_workers=num_workers, shuffle=True,
                            batch_size=Config.BATCH_SIZE)

    model = Transformer(
        Config.SRC_VOCAB_SIZE,
        Config.TRG_VOCAB_SIZE,
        Config.HIDDEN_SIZE, 
        Config.NUM_LAYERS,
    ).to(Config.DEVICE)

    optimizer = optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PADDING_IDX)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number Parameters:", pytorch_total_params)

    best_valid_loss = float('inf')

    for epoch in range(Config.NUM_EPOCHS):
        print(f'Epoch: {epoch+1:02}')

        train_loss = train(model, optimizer, criterion, train_loader)
        valid_loss = evaluate(model, criterion, val_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val   Loss: {valid_loss:.3f}')

    print(best_valid_loss)

if __name__ == '__main__':
    run_sanity_check()
    