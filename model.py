"""
Contains all models used in the experiment
"""

import math
import torch
import torch.nn as nn
from config import Config


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=500):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.scale = nn.Parameter(torch.ones(1))

#         pe = torch.zeros(Config.BATCH_SIZE, max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(math.log(10000.0) / d_model))
#         pe[:, :, 0::2] = torch.sin(position * div_term)
#         pe[:, :, 1::2] = torch.cos(position * div_term)

#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.scale * self.pe[:, :x.size(1)]
#         return x

# class Transformer(nn.Module):
#     def __init__(
#         self,
#         embedding_size, 
#         vocab_size,
#         d_model,
#         nhead,
#         dim_feedforward,
#         dropout,
#         device
#     ):
#         #embedding
#         super(Transformer, self).__init__()
#         self.d_model = d_model
#         self.device = device
#         self.src_word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=Config.PADDING_IDX)
#         self.src_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        
#         self.trg_word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=Config.PADDING_IDX)
#         self.trg_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        
#         #transformer
#         self.transformer = nn.Transformer(
#             d_model=d_model, 
#             nhead=nhead,
#             num_encoder_layers=Config.NUM_LAYERS, 
#             num_decoder_layers=Config.NUM_LAYERS, 
#             dim_feedforward=dim_feedforward, 
#             dropout=dropout, 
#             batch_first=True, 
#             device=Config.DEVICE
#         )
            
#         self.linear = nn.Linear(in_features=embedding_size, out_features=vocab_size)
#         self.dropout = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
    
#     def src_mask(self, src):
#         src_mask = src == Config.PADDING_IDX
#         return src_mask

#     def make_trg_mask(self, trg):
#         trg_mask = trg != Config.PADDING_IDX
#         return trg_mask

#     def generate_square_subsequent_mask(self, size): # Generate mask covering the top right triangle of a matrix
#         mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def forward(self, src, trg):
#         batch_size, src_seq_length  = src.shape
#         batch_size, trg_seq_length = trg.shape
        
#         #source and target position    
#         embed_src = self.dropout((self.src_word_embedding(src)))
#         embed_src = self.dropout(self.src_pos_encoder(embed_src))

#         embed_trg = self.dropout((self.trg_word_embedding(trg)))
#         embed_trg = self.dropout(self.trg_pos_encoder(embed_src))
        
#         #source and target mask
#         src_key_padding_mask = self.src_mask(src)
#         trg_key_padding_mask = self.src_mask(trg)
#         trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(Config.DEVICE)
        
#         out = self.transformer(
#             src=embed_src, 
#             tgt=embed_trg,
#             src_mask=None,
#             tgt_mask=trg_mask, 
#             memory_mask=None,
#             src_key_padding_mask=src_key_padding_mask,
#             tgt_key_padding_mask=trg_key_padding_mask,
#             memory_key_padding_mask=src_key_padding_mask 
#         )

#         out = self.linear(out)
        
#         return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, hidden, num_layers=5, dropout=0.1): #embedding_size, vocab_size,
        super(Transformer, self).__init__()
        nhead = hidden//64
        
        self.encoder = nn.Embedding(src_vocab_size, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Embedding(trg_vocab_size, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=hidden*4, dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(hidden, trg_vocab_size)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
            
        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output
