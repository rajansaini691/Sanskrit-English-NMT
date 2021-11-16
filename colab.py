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
from preprocess import preprocess_data
from config import Config

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

def create_config():
    config = dict()

    config['data_dir'] = 'output'

    config['dataset_limit'] = None
    config["dataset_limit"] = None
    config["print_every"] = 1
    config["save_every"] = 1

    config["vocabulary_size"] = Config.VOCAB_SIZE
    config["share_dictionary"] = False
    config["positional_encoding"] = True

    config["d_model"] = Config.D_MODEL
    config["layers_count"] = Config.NUM_LAYERS
    config["heads_count"] = Config.NUM_HEADS
    config["d_ff"] = Config.FEED_FORWARD_DIM
    config["dropout_prob"] = Config.DROPOUT

    config["label_smoothing"] = 0.1
    config["optimizer"] = "Adam"
    config["lr"] = Config.LEARNING_RATE
    config["clip_grads"] = True

    config["batch_size"] = Config.BATCH_SIZE
    config["epochs"] = Config.NUM_EPOCHS

    with open(os.path.join(Config.OUT_DIR, "transformer_config.json"), "w") as outfile:
      json.dump(config, outfile)


def one_epoch(model, dataloader, running_loss, writer, loss_function, epoch, start_batch, optimizer, train):
    if train == True:
      model.train()
    else:
      model.eval()
    
    update = 0
    number_exeptions = 0
    loss = 0

    for index in range(start_batch, 200, Config.BATCH_SIZE):

        if train:
          print(f"[Training Epoch] {epoch}/{Config.NUM_EPOCHS - 1}, Batch Number: {index}/{len(dataloader[0])}")
        else:
          print(f"[Validation Epoch] {epoch}/{Config.NUM_EPOCHS - 1}, Validating: {index}/{len(dataloader[0])}")
        
        try:
          loss = 0
          input, target = get_batch(index, dataloader)

          # extra_input_padding = torch.full((Config.BATCH_SIZE, 1), Config.PADDING_IDX, dtype=int)
          bos_target_padding = torch.full((Config.BATCH_SIZE, 1), 1, dtype=int)

          # model_input = torch.cat((input, extra_input_padding), dim=1)
          model_target = torch.cat((bos_target_padding, target), dim=1)[:, :-1]
 
          model_output = model(input.to(Config.DEVICE), model_target.to(Config.DEVICE)).to(Config.DEVICE)
            
          loss_output = model_output.reshape(-1, model_output.shape[2])
          loss_target = target.reshape(-1)

          loss = (loss_function(loss_output.to(Config.DEVICE), loss_target.to(Config.DEVICE)))

          if train:
            optimizer.zero_grad()
            loss.backward()        
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
        
        #update tensorboard and save model
        if update == 10:    # every 10 mini-batches
            running_avg = running_loss / 10
            graph = ''
            if train:
              checkpoint = {
                  "epoch":epoch,
                  "batch":index,
                  "model_state":model.state_dict(),
                  "optim_state":optimizer.state_dict()
              }
              torch.save(checkpoint, os.path.join(Config.OUT_DIR, Config.CHECKPOINT_PATH))
              graph = 'training loss'
            else:
              graph = 'validation loss'
            writer.add_scalar(graph,
                            running_avg,
                            epoch * len(dataloader) + index)
            print(f"[Loss] {running_avg}")
            running_loss = 0.0

            update = 0

# Initialize out dir
if not os.path.exists(Config.OUT_DIR):
    os.path.mkdir(Config.OUT_DIR)

print('Device:', Config.DEVICE)

dataset = load_dataset("rahular/itihasa")

training_data = dataset['train']
validation_data = dataset['validation']
test_data = dataset['test']

preprocess_data(training_data, validation_data, test_data)

model = Transformer(
    Config.EMBEDDING_SIZE,
    Config.VOCAB_SIZE,
    Config.D_MODEL,
    Config.NUM_HEADS,
    Config.FEED_FORWARD_DIM,
    Config.DROPOUT,
    Config.DEVICE,
).to(Config.DEVICE)

create_config()

optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
loss_function = nn.CrossEntropyLoss(ignore_index=Config.PADDING_IDX)
start_epoch = 0
start_batch = 0

if Config.LOAD_MODEL:
  checkpoint = torch.load(os.path.join(Config.OUT_DIR, Config.CHECKPOINT_PATH), map_location=Config.DEVICE)

  start_batch = checkpoint["batch"]
  start_epoch = checkpoint["epoch"]  
  model.load_state_dict(checkpoint["model_state"])
  optimizer.load_state_dict(checkpoint["optim_state"])

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number Parameters:", pytorch_total_params)

#tensorboard
writer = SummaryWriter("runs")

running_loss = 0.0
running_val_loss = 0.0

for epoch in range(start_epoch, Config.NUM_EPOCHS):
    print(f"[Epoch] {epoch}/{Config.NUM_EPOCHS - 1}")
    
    training = (torch.load(os.path.join(Config.OUT_DIR, 'train_data_eng.pth')), torch.load(os.path.join(Config.OUT_DIR, 'train_data_san.pth'))) #shuffled_copies(training_dataloader[0], training_dataloader[1])
    validation = (torch.load(os.path.join(Config.OUT_DIR, 'valid_data_eng.pth')), torch.load(os.path.join(Config.OUT_DIR, 'valid_data_san.pth')))
    
    one_epoch(model, training, running_loss, writer, loss_function, epoch, start_batch, optimizer, train=True)
    one_epoch(model, validation, running_val_loss, writer, loss_function, epoch, start_batch, optimizer, train=False)
    
    start_batch = 0
    
