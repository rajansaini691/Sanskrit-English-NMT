import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import sys
import math
import random
import subword_nmt
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.get_vocab import get_vocab
from subword_nmt.apply_bpe import BPE
from torch.autograd import Variable
from itertools import chain
from string import ascii_lowercase, ascii_uppercase
from datasets import load_dataset

class Config():
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 10
    VOCAB_SIZE = 10735
    EMBEDDING_SIZE = 512
    D_MODEL = 512
    NUM_HEADS = 8
    FEED_FORWARD_DIM = 1024
    NUM_LAYERS = 6
    DROPOUT = 0.1
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PADDING_IDX = 10732
    OUT_DIR = 'output'
    CHECKPOINT_PATH = 'checkpoint_200_samples.pth'
    LOAD_MODEL = False

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(Config.BATCH_SIZE, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(
        #     0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:, :x.size(1)]
        return x
  
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size, 
        vocab_size,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        device
    ):
        #embedding
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.device = device
        self.src_word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=Config.PADDING_IDX)
        self.src_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        self.trg_word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=Config.PADDING_IDX)
        self.trg_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        #transformer
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead,
            num_encoder_layers=Config.NUM_LAYERS, 
            num_decoder_layers=Config.NUM_LAYERS, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True, 
            device=Config.DEVICE
        )
            
        self.linear = nn.Linear(in_features=embedding_size, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def src_mask(self, src):
        src_mask = src == Config.PADDING_IDX
        return src_mask

    def make_trg_mask(self, trg):
        trg_mask = trg != Config.PADDING_IDX
        return trg_mask

    def generate_square_subsequent_mask(self, size): # Generate mask covering the top right triangle of a matrix
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg):
        batch_size, src_seq_length  = src.shape
        batch_size, trg_seq_length = trg.shape
        
        # src_pe = torch.zeros(batch_size, src_seq_length, self.d_model)
        # src_position = torch.arange(0, src_seq_length).unsqueeze(1)
        # src_div_term = torch.exp(torch.arange(0, self.d_model, 2) *
        #                      -(math.log(10000.0) / self.d_model))
        # src_pe[:, :, 0::2] = torch.sin(src_position * src_div_term)
        # src_pe[:, :, 1::2] = torch.cos(src_position * src_div_term)

        # trg_pe = torch.zeros(batch_size, src_seq_length, self.d_model)
        # trg_position = torch.arange(0, src_seq_length).unsqueeze(1)
        # trg_div_term = torch.exp(torch.arange(0, self.d_model, 2) *
        #                      -(math.log(10000.0) / self.d_model))
        # trg_pe[:, :, 0::2] = torch.sin(trg_position * trg_div_term)
        # trg_pe[:, :, 1::2] = torch.cos(trg_position * trg_div_term)
        
        #source and target position    
        embed_src = self.dropout((self.src_word_embedding(src)))
        embed_src = self.dropout(self.src_pos_encoder(embed_src))

        embed_trg = self.dropout((self.trg_word_embedding(trg)))
        embed_trg = self.dropout(self.trg_pos_encoder(embed_src))
        
        #source and target mask
        src_key_padding_mask = self.src_mask(src)
        trg_key_padding_mask = self.src_mask(trg)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(Config.DEVICE)
        
        out = self.transformer(
            src=embed_src, 
            tgt=embed_trg,
            src_mask=None,
            tgt_mask=trg_mask, 
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask 
        )

        out = self.linear(out)
        
        return out


def generate_vocab(src_file, trg_file):
    """
    Uses byte-pair encoding to build a vocabulary from the corpus
    Returns a file object pointing to the generated codes file
    """
    path_to_codes_file_eng = "codes_file_eng"
    path_to_codes_file_san = "codes_file_san"

    if not os.path.exists(Config.OUT_DIR):
        os.makedirs(Config.OUT_DIR)

    if os.path.exists(os.path.join(Config.OUT_DIR, path_to_codes_file_eng)) \
        and os.path.exists(os.path.join(Config.OUT_DIR, path_to_codes_file_san)):
        codes_file_eng = open(os.path.join(Config.OUT_DIR, path_to_codes_file_eng), "r")
        codes_file_san = open(os.path.join(Config.OUT_DIR, path_to_codes_file_san), "r")
        return codes_file_eng, codes_file_san

    if os.path.exists(os.path.join(Config.OUT_DIR, 'eng_train')) \
      and os.path.exists(os.path.join(Config.OUT_DIR, 'san_train')):
      codes_file_eng = open(os.path.join(Config.OUT_DIR, path_to_codes_file_eng), "w")
      codes_file_san = open(os.path.join(Config.OUT_DIR, path_to_codes_file_san), "w")

      learn_bpe(src_file, codes_file_eng, 10000)
      learn_bpe(trg_file, codes_file_san, 10000)
    
      return codes_file_eng, codes_file_san
    
    else:
      raise 'No eng_train and san_train file'

def generate_data_file(data, dtype):

    if not os.path.exists(Config.OUT_DIR):
        os.makedirs(Config.OUT_DIR)
    
    eng = open(os.path.join(Config.OUT_DIR, f'eng_{dtype}'), 'w')
    san = open(os.path.join(Config.OUT_DIR, f'san_{dtype}'), 'w')

    for translation in data['translation']:
      eng.write(translation['en'] + '\n')
      san.write(translation['sn'] + '\n')
    
    eng.close()
    san.close()

    eng = open(os.path.join(Config.OUT_DIR, f'eng_{dtype}'), 'r')
    san = open(os.path.join(Config.OUT_DIR, f'san_{dtype}'), 'r')

    return eng, san


def tokenize_dataset(infile, codes_file, prefix=""):
    """
    Use a vocabulary to tokenize the dataset
    """
    path_to_tokenized_file = os.path.join(Config.OUT_DIR, prefix + "_tokenized")
    if not os.path.exists(Config.OUT_DIR):
        os.makedirs(Config.OUT_DIR)

    if os.path.exists(path_to_tokenized_file):
        with open(path_to_tokenized_file) as codes_file:
            return codes_file
    tokenized_corpus = open(path_to_tokenized_file, "w")
    codes_file = open(codes_file.name, "r")
    infile.seek(0)
    codes_file.seek(0)

    bpe = BPE(codes_file)
    for line in infile:
        tokenized_line = bpe.process_line(line)
        tokenized_corpus.writelines(tokenized_line)
    return tokenized_corpus

def create_token_dict(codes_file, dtype):
    """
    Creates a dictionary mapping subwords to integers
    """
    codes_file = open(codes_file.name, "r")
    codes_file.seek(0)
    next(codes_file)    # Skip first line (contains version info)

    token_dict = dict()

    # Insert alphabet first
    for i, x in enumerate(chain(ascii_lowercase, ascii_uppercase)):
        token_dict[x] = i
    for i, x in enumerate(chain(ascii_lowercase, ascii_uppercase)):
        token_dict[x + '</w>'] = i

    alphabet_len = len(token_dict.keys())

    with open(os.path.join(Config.OUT_DIR, f'vocabulary-{dtype}.txt'), 'w') as vocab:
      # Insert bpe subwords
      for i, line in enumerate(codes_file):
          token = line.replace(' ', '').replace('\n', '')
          token_dict[token] = i + alphabet_len

          vocab.write(f'{i+alphabet_len}\t{token}\t0\n')

    return token_dict
    
def create_tensor_from_sentence(sentence, token_dict):
    sentence_with_boundary_tokens = list(map(\
        lambda token: token[:-2] if token.endswith("@@") else token + "</w>",
        sentence.replace('\n', '').split(' ')))
    sentence_as_indices = []
    for token in sentence_with_boundary_tokens:
        try:
            sentence_as_indices += [token_dict[token]]
        except KeyError as e:
            pass
    return torch.tensor(sentence_as_indices, dtype=torch.long)

def create_tensors(tokenized_corpus_file, token_dict):
    """
    Create pytorch-processable dataset from corpus
    (return list of seq2seq tensors)
    """
    tokenized_corpus_file = open(tokenized_corpus_file.name)
    data = []
    for line in tokenized_corpus_file:
        try:
            data.append(create_tensor_from_sentence(line, token_dict))
        except ValueError:
            pass
    return data

# TODO Get dataset from parsed args
def preprocess_data(train_data, valid_data, test_data):
    # Use cached data if already computed
    if os.path.exists(os.path.join(Config.OUT_DIR, "train_data")) and os.path.exists(os.path.join(Config.OUT_DIR, "valid_data")) \
        and os.path.exists(os.path.join(Config.OUT_DIR, "vocab_size.npy")):
        print("Loading cached train/val data...")
        return [torch.load(Config.OUT_DIR + "train_data"), torch.load(Config.OUT_DIR + "val_data"), np.load(Config.OUT_DIR + "vocab_size.npy")]

    # Run BPE tokenization
    print("Generating vocabulary...")
    eng_train, san_train = generate_data_file(train_data, dtype="train")
    eng_valid, san_valid = generate_data_file(valid_data, dtype="valid")
    eng_test, san_test = generate_data_file(test_data, dtype="test")
    
    codes_file_eng, codes_file_san = generate_vocab(eng_train, san_train)
    
    print("Tokenizing data...")
    tokenized_train_set_eng = tokenize_dataset(eng_train, codes_file_eng, prefix="eng_train")
    tokenized_train_set_san = tokenize_dataset(san_train, codes_file_san, prefix="san_train")

    tokenized_valid_set_eng = tokenize_dataset(eng_valid, codes_file_eng, prefix="eng_valid")
    tokenized_valid_set_san = tokenize_dataset(san_valid, codes_file_san, prefix="san_valid")

    tokenized_test_set_eng = tokenize_dataset(eng_test, codes_file_eng, prefix="eng_test")
    tokenized_test_set_san = tokenize_dataset(san_test, codes_file_san, prefix="san_test")

    # Map tokens to index in vocabulary
    print("Creating token dictionary...")
    eng_token_dict = create_token_dict(codes_file_eng, dtype='source')
    san_token_dict = create_token_dict(codes_file_san, dtype='target')


    # Create tensors for each set of parallel sentences
    print("Creating tensors...")
    train_data_eng = create_tensors(tokenized_train_set_eng, eng_token_dict)
    train_data_san = create_tensors(tokenized_train_set_san, san_token_dict)

    valid_data_eng = create_tensors(tokenized_valid_set_eng, eng_token_dict)
    valid_data_san = create_tensors(tokenized_valid_set_san, san_token_dict)

    test_data_eng = create_tensors(tokenized_test_set_eng, eng_token_dict)
    test_data_san = create_tensors(tokenized_test_set_san, san_token_dict)
    
    # Cache tensors and vocab size
    torch.save(train_data_eng, os.path.join(Config.OUT_DIR, "train_data_eng.pth"))
    torch.save(train_data_san, os.path.join(Config.OUT_DIR, "train_data_san.pth"))

    torch.save(valid_data_eng, os.path.join(Config.OUT_DIR, "valid_data_eng.pth"))
    torch.save(valid_data_san, os.path.join(Config.OUT_DIR, "valid_data_san.pth"))

    torch.save(test_data_eng, os.path.join(Config.OUT_DIR, "test_data_eng.pth"))
    torch.save(test_data_san, os.path.join(Config.OUT_DIR, "test_data_san.pth"))
    
    # torch.save(val_data, OUT_DIR + "val_data")
    np.save(os.path.join(Config.OUT_DIR, "eng_vocab_size"), len(eng_token_dict))
    np.save(os.path.join(Config.OUT_DIR, "san_vocab_size"), len(san_token_dict))

    return train_data_eng, train_data_san, valid_data_eng, valid_data_san, test_data_eng, test_data_san

def get_batch(batch_num, dataloader):
  data_batch = (dataloader[0][batch_num: (batch_num+Config.BATCH_SIZE)], dataloader[1][batch_num: (batch_num+Config.BATCH_SIZE)])
  
  max_length = 0
  for sequence in range(len(data_batch[0])):
    if len(data_batch[0][sequence]) > max_length:
      max_length = len(data_batch[0][sequence])
    if len(data_batch[1][sequence]) > max_length:
      max_length = len(data_batch[1][sequence])
  
  new_x = []
  for sequence in range(len(data_batch[0])):
    new_seq = []  
    for word in data_batch[0][sequence]:
      new_seq.append(word)
    while len(new_seq) < max_length:
      new_seq.append(Config.PADDING_IDX)
    new_x.append(new_seq)

  new_y = []
  for sequence in range(len(data_batch[1])):
    new_seq = []  
    for word in data_batch[1][sequence]:
      new_seq.append(word)
    while len(new_seq) < max_length:
      new_seq.append(Config.PADDING_IDX)
    new_y.append(new_seq)
    np.array(new_y).shape

  new_x = torch.as_tensor(new_x, dtype=int)
  new_y = torch.as_tensor(new_y, dtype=int)

  return (new_x, new_y)

def shuffled_copies(a, b):
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
          print(model_target)
          print(input)
 
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
    
