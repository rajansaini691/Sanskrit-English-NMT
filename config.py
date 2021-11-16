import torch

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

