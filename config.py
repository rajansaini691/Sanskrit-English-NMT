import torch

class Config():
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.01
    BATCH_SIZE = 10
    SRC_VOCAB_SIZE = 11000
    TRG_VOCAB_SIZE = 11000
    EMBEDDING_SIZE = 512
    HIDDEN_SIZE = 512
    D_MODEL = 512
    NUM_HEADS = 8
    FEED_FORWARD_DIM = 1024
    NUM_LAYERS = 3
    DROPOUT = 0.1
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR = 'output'
    DATA_DIR = 'data'
    DRIVE_PATH = 'drive/MyDrive'
    PRETRAINED_CHECKPOINT_PATH = 'pretrained_checkpoint.pth'
    FINE_TUNED_CHECKPOINT_PATH = 'checkpoint.pth'
    LOAD_MODEL = True
    MULTILINGUAL = True
    PADDING_IDX = 10999
    SAN_TOKEN = 10998
    PLI_TOKEN = 10997
    UNK_TOKEN = 10996
    BOS_TOKEN = 1
    EOS_TOKEN = 0
    
