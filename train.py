"""
Original code copy and pasted from the colab
"""

import os
import json

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datasets import load_dataset
from model import Transformer
from preprocess import convert_itihasa_dataset_to_tensors, load_marathi_dataset
from config import Config

def get_batch_seq(batch_num, dataloader):
    '''
    Takes in a data loader and outputs a tuple (src, trg) of shape [sequence, batch]
    '''
    
    features = dataloader[0][batch_num: (batch_num+Config.BATCH_SIZE)]
    labels = dataloader[1][batch_num: (batch_num+Config.BATCH_SIZE)]
    
    labels = [torch.cat((torch.tensor([1]), label)) for label in labels] #add bos token (1)
    labels = [torch.cat((label, torch.tensor([0]))) for label in labels] #add eos token (0)
    
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=Config.PADDING_IDX)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=Config.PADDING_IDX)
    
    source = None
    for src in features:
        if type(source) == type(None):
            source = np.array([src.numpy()]).T
        else:
            source = np.concatenate((source, np.array([src.numpy()]).T), axis=1)  
    
        
    target = None
    for trg in labels:
        if type(target) == type(None):
            target = np.array([trg.numpy()]).T
        else:
            target = np.concatenate((target, np.array([trg.numpy()]).T), axis=1)
    
    source = torch.as_tensor(source)
    target = torch.as_tensor(target)
    
    return source.to(Config.DEVICE), target.to(Config.DEVICE)

def get_batch(batch_num, dataloader):
    """
    Pulls a padded batch from the dataloader

    Parameters:
        batch_num       Batch will start at this index; should be a multiple of the BATCH_SIZE
        dataloader      Dataloader for the dataset. Dataset consists of (src, tgt), where
                        src[i] translates to tgt[i]
    """
    data_batch = (
            dataloader[0][batch_num: (batch_num+Config.BATCH_SIZE)],
            dataloader[1][batch_num: (batch_num+Config.BATCH_SIZE)])
    # Calculate the max sentence length over the entire batch
    max_length = 0
    for sequence in range(len(data_batch[0])):
        if len(data_batch[0][sequence]) > max_length:
            max_length = len(data_batch[0][sequence])
        if len(data_batch[1][sequence]) > max_length:
            max_length = len(data_batch[1][sequence])

    # Pad the ends of each source sentence to the max length
    new_x = []
    for sequence in range(len(data_batch[0])):
        new_seq = []
        for word in data_batch[0][sequence]:
            new_seq.append(word)
        while len(new_seq) < max_length:
            new_seq.append(Config.PADDING_IDX)
        new_x.append(new_seq)

    # Pad the ends of each target sentence to the max length
    new_y = []
    for sequence in range(len(data_batch[1])):
      new_seq = []  
      for word in data_batch[1][sequence]:
        new_seq.append(word)
      while len(new_seq) < max_length:
        new_seq.append(Config.PADDING_IDX)
      new_y.append(new_seq)
      np.array(new_y).shape

    # Convert to pytorch tensors for training
    new_x = torch.as_tensor(new_x, dtype=int)
    new_y = torch.as_tensor(new_y, dtype=int)

    return (new_x, new_y)

def shuffled_copies(a, b):
    """
    Shuffle a and b such that a[i] and b[i] are
    paired after the shuffle
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def one_epoch(model, dataloader, writer, criterion, epoch, start_batch, optimizer, train):
    """
    Run the model through a single pass through the dataset defined by the dataloader

    model - The model being trained. Inherits torch.nn.Module
    dataloader - Encodes the dataset
    writer - SummaryWriter for Tensorboard
    loss_function - Pytorch loss function, like cross entropy
    epoch - Current epoch number, for printing status
    start_batch - (integer) Where to start the epoch (indexes the dataset)
    optimizer - Pytorch optimizer (like Adam)
    train - If set to True: will train on the dataset, update parameters, and save checkpoints.
            Otherwise, will run model over dataset without training and report loss
            (used for validation)
    """
    if train == True:
      model.train()
    else:
      model.eval()
    
    update = 0
    number_exeptions = 0
    loss = 0
    running_loss = 0

    for index in range(start_batch, len(dataloader[0]), Config.BATCH_SIZE):
        if train:
            print(f"[Training Epoch] {epoch}/{Config.NUM_EPOCHS - 1}, Batch Number: {index}/{len(dataloader[0])}")
        else:
            print(f"[Validation Epoch] {epoch}/{Config.NUM_EPOCHS - 1}, Validating: {index}/{len(dataloader[0])}")
        
        try:
            loss = 0
            source, target = get_batch_seq(index, dataloader)
            
            # print('src', source)
            # print('trg', target)
                
            output = model(source, target[:-1,:])

            loss = criterion(output.transpose(0, 1).transpose(1, 2), target[1:,:].transpose(0, 1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        except Exception as e:
            number_exeptions += 1
            print('[EXCEPTION]', e)
            print('Memory', torch.cuda.memory_allocated(Config.DEVICE))
            print('Number Exceptions', number_exeptions)
            torch.cuda.empty_cache()
            continue

        update += 1
        running_loss += loss.item()

        # update tensorboard and save model
        if update == 100:    # every 10 mini-batches
            running_avg = running_loss / 100
            graph = ''
            if train:
                checkpoint = {
                    "epoch":epoch,
                    "batch":index,
                    "model_state":model.state_dict(),
                    "optim_state":optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(Config.DRIVE_PATH, Config.CHECKPOINT_PATH))
                graph = 'training loss'
            else:
                graph = 'validation loss'
            
            writer.add_scalar(graph,
                            running_avg,
                            epoch * len(dataloader) + index)
            print(f"[Loss] {running_avg}")
            running_loss = 0.0

            update = 0

def train():
    # Initialize out dir
    if not os.path.exists(Config.OUT_DIR):
        os.mkdir(Config.OUT_DIR)

    print('Device:', Config.DEVICE)

    # dataset = load_dataset("rahular/itihasa")

    # training_data = dataset['train']
    # validation_data = dataset['validation']
    # test_data = dataset['test']
    
    eng_train, mar_train=load_marathi_dataset(os.path.join(Config.DATA_DIR, "en-mr"))
    # convert_itihasa_dataset_to_tensors(training_data, validation_data, test_data)

    # convert_itihasa_dataset_to_tensors(training_data, validation_data, test_data)
    # load_marathi_dataset(os.path.join(Config.DATA_DIR, "en-mr"))

    model = Transformer(
        Config.SRC_VOCAB_SIZE,
        Config.TRG_VOCAB_SIZE,
        Config.HIDDEN_SIZE, 
        Config.NUM_LAYERS,
    ).to(Config.DEVICE)

    optimizer = optim.AdamW(model.parameters())
    loss_function = nn.CrossEntropyLoss(ignore_index=Config.PADDING_IDX)
    start_epoch = 0
    start_batch = 0

    if Config.LOAD_MODEL:
        checkpoint = torch.load(os.path.join(Config.OUT_DIR, Config.CHECKPOINT_PATH),
                                map_location=Config.DEVICE)

        start_batch = checkpoint["batch"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number Parameters:", pytorch_total_params)

    # Tensorboard
    writer = SummaryWriter("runs")

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"[Epoch] {epoch}/{Config.NUM_EPOCHS - 1}")

        # TODO Use the returned values from convert_itihasa_dataset_to_tensors() rather
        # than loading from disk like this
        training = (eng_train, mar_train)
        # validation = (torch.load(os.path.join(Config.OUT_DIR, 'itihasa_eng_val.pth')),
                    # torch.load(os.path.join(Config.OUT_DIR, 'itihasa_san_val.pth')))

        one_epoch(model, training, writer, loss_function, epoch, start_batch, optimizer, train=True)
        # one_epoch(model, validation, writer, loss_function, epoch, start_batch, optimizer, train=False)

        start_batch = 0

if __name__ == '__main__':
    train()
