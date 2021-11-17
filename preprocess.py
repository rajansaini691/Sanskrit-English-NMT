import os
import torch
from config import Config
from itertools import chain
from string import ascii_lowercase, ascii_uppercase
import numpy as np
import subword_nmt
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

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


