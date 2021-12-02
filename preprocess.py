import os
import torch
from config import Config
from itertools import chain
from string import ascii_lowercase, ascii_uppercase
import numpy as np
import subword_nmt
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

def generate_vocab(corpus_file, path_to_codes_file):
    """
    Uses byte-pair encoding to build a vocabulary from the corpus

    Returns a file object pointing to the generated codes file. If the codes
    file already exists, we assume it was precomputed, so we exit early. 

    Parameters:
        corpus_file         File-like object pointing to monolingual corpus
        path_to_codes_file  Location to write the codes file
    """
    # Check if codes file is cached
    if os.path.exists(path_to_codes_file):
        return open(path_to_codes_file, "r")

    # Generate vocabulary
    codes_file = open(path_to_codes_file, "w")
    learn_bpe(corpus_file, codes_file, 10000)
    return codes_file
    
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

    Parameters:
        infile      Path to the corpus getting tokenized
        codes_file  Path to the vocabulary

    Returns a file object pointing to the tokenized corpus
    """
    path_to_tokenized_file = os.path.join(Config.OUT_DIR, prefix + "_tokenized")
    if not os.path.exists(Config.OUT_DIR):
        os.makedirs(Config.OUT_DIR)

    if os.path.exists(path_to_tokenized_file):
        with open(path_to_tokenized_file, 'r') as tokenized_file:
            return tokenized_file

    tokenized_corpus = open(path_to_tokenized_file, "w")
    codes_file = open(codes_file.name, "r")
    infile.seek(0)
    codes_file.seek(0)

    bpe = BPE(codes_file)
    for line in infile:
        tokenized_line = bpe.process_line(line)
        tokenized_corpus.writelines(tokenized_line)
    return tokenized_corpus

# TODO Rename dtype to prefix, document parameters
def create_token_dict(codes_file, dtype):
    """
    Creates a dictionary mapping subwords to integers
    """
    codes_file = open(codes_file.name, "r")
    codes_file.seek(0)
    next(codes_file)    # Skip first line (contains version info)

    token_dict = dict()
    
    token_dict['<EOS>'] = Config.EOS_TOKEN
    token_dict['<BOS>'] = Config.BOS_TOKEN

    # Insert alphabet first
    if dtype == 'source':
        for i, x in enumerate(chain(ascii_lowercase, ascii_uppercase)):
            token_dict[x] = len(token_dict)
            token_dict[x + '</w>'] = len(token_dict)
    
    elif dtype == 'target':
        for i, x in enumerate(range(0x0900, 0x097F)): #sanksrit unicode characters
            token_dict[chr(x)] = len(token_dict)
            token_dict[chr(x) + '</w>'] = len(token_dict)

    alphabet_len = len(token_dict.keys())

    # Insert bpe subwords
    for i, line in enumerate(codes_file):
        token = line.replace(' ', '').replace('\n', '')
        token_dict[token] = i + alphabet_len

    token_dict['<SAN>'] = Config.SAN_TOKEN
    token_dict['<PLI>'] = Config.PLI_TOKEN

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
    tokenized_corpus_file = open(tokenized_corpus_file.name, 'r')
    data = []
    for line in tokenized_corpus_file:
        try:
            data.append(create_tensor_from_sentence(line, token_dict))
        except ValueError:
            pass
    return data

# TODO Accept the whole itihasa json and do the train/val/test split inside here
# TODO Accept a vocab parameter and create the codes_file externally.
#      This way, we can use monolingual data for the vocab.
# TODO Add an out_dir parameter
# TODO Document what gets returned
def convert_itihasa_dataset_to_tensors(itihasa_training_data, itihasa_validation_data, itihasa_test_data):
    """
    Converts the itihasa dataset into a list of tensors of subword tokens that can
    be fed into a pytorch model.
    """
    eng_train, san_train = generate_data_file(itihasa_training_data, dtype="train")
    eng_val, san_val = generate_data_file(itihasa_validation_data, dtype="val")
    eng_test, san_test = generate_data_file(itihasa_test_data, dtype="test")

    print("Generating vocabulary...")
    codes_file_eng = generate_vocab(eng_train, os.path.join(Config.OUT_DIR, "codes_file_eng"))
    codes_file_san = generate_vocab(san_train, os.path.join(Config.OUT_DIR, "codes_file_san"))

    return [
            corpus_to_tensors(eng_train, codes_file_eng, Config.OUT_DIR, "itihasa_eng_train"),
            corpus_to_tensors(san_train, codes_file_san, Config.OUT_DIR, "itihasa_san_train"),
            corpus_to_tensors(eng_val, codes_file_eng, Config.OUT_DIR, "itihasa_eng_val"),
            corpus_to_tensors(san_val, codes_file_san, Config.OUT_DIR, "itihasa_san_val"),
            corpus_to_tensors(eng_test, codes_file_eng, Config.OUT_DIR, "itihasa_eng_test"),
            corpus_to_tensors(san_test, codes_file_san, Config.OUT_DIR, "itihasa_san_test")]

# TODO Get dataset from parsed args
def corpus_to_tensors(corpus_file, vocab, out_dir, prefix):
    """
    Converts a corpus into a list of tensors. This corpus should contain a set of sentences
    in the same language separated by newlines.

    Returns a list of tensors of tokens:
        For example, suppose corpus_file points to corpus.txt, which contains
        "Hello!\nThis is an English sentence.\nI am studying Machine Translation\n".
        This function should then return something like:
        [Tensor([103, 202]), Tensor([101, ..., 304]), Tensor([302, ..., 109])]

    Parameters:
        corpus_file     File object pointing to the corpus
        vocab           File object pointing to a vocabulary generated by the
                        subword_nmt library for this language. This vocabulary
                        is used to tokenize the dataset into subwords.
        out_dir         Directory to emit cached results (note that the
                        generated files are not necessarily human-readable)
        prefix          Prefix to add to all generated files:
                         - For example, setting prefix="itihasa_eng_train"
                           will cause {out_dir}/itihasa_eng_train_tokenized.txt
                           to be generated, along with others
                         - If you are calling this function multiple times on
                           different datasets, you MUST use unique prefixes
    """
    path_to_cached_tensors = os.path.join(out_dir, f"{prefix}.pth")
    if os.path.exists(path_to_cached_tensors):
        print(f"Loading cached tensors for {prefix}...")
        return torch.load(path_to_cached_tensors)

    # Use the vocabulary to split each sentence into a list of subwords, where
    # each subword is an element of the vocabulary
    print(f"Tokenizing {prefix}...")
    tokenized_dataset = tokenize_dataset(corpus_file, vocab, prefix=prefix)

    # Map tokens to index in vocabulary
    print(f"Creating token dictionary from vocab associated with {prefix}...")
    token_dict = create_token_dict(vocab, dtype=prefix)

    # Turn tokenized dataset into a list of tensors of integers, where each integer
    # maps to a token
    print(f"Creating tensors for {prefix}...")
    tensors = create_tensors(tokenized_dataset, token_dict)

    # Cache tensors and vocab size
    torch.save(tensors, os.path.join(Config.OUT_DIR, f"{prefix}.pth"))

    return tensors

# FIXME Should Sanskrit and Marathi share a vocabulary? They're separate right now.
def load_marathi_dataset(path_to_en_mr_dir):
    """
    Loads an english-marathi parallel dataset into two lists of aligned token vectors.
    
    Parameters:
        path_to_en_mr_dir       Root of the english-marathi text. It should contain
                                the files train.mr and train.en.
    """
    with open(os.path.join(path_to_en_mr_dir, "train.en"), "r") as eng_train_file:
        with open(os.path.join(path_to_en_mr_dir, "train.mr"), "r") as mr_train_file:
            print("Generating vocabulary for en-mr English...")
            codes_file_eng = generate_vocab(eng_train_file, os.path.join(Config.OUT_DIR, "codes_file_eng"))
            print("Generating vocabulary for en-mr Marathi...")
            codes_file_mr = generate_vocab(mr_train_file, os.path.join(Config.OUT_DIR, "codes_file_mr"))

            return (corpus_to_tensors(eng_train_file, codes_file_eng, Config.OUT_DIR, "samanatar_en_mr_english_full"),
                    corpus_to_tensors(mr_train_file, codes_file_mr, Config.OUT_DIR, "samanatar_en_mr_marathi_full"))

def load_pali_dataset(path_to_en_pali_dir):
    """
    Loads an english-marathi parallel dataset into two lists of aligned token vectors.
    
    Parameters:
        path_to_en_mr_dir       Root of the english-marathi text. It should contain
                                the files train.mr and train.en.
    """
    with open(os.path.join(path_to_en_pali_dir, "train.en"), "r") as eng_train_file:
        with open(os.path.join(path_to_en_pali_dir, "train.pli"), "r") as pali_train_file:
            print("Generating vocabulary for en-pali English...")
            codes_file_eng = generate_vocab(eng_train_file, os.path.join(Config.OUT_DIR, "codes_file_eng"))
            print("Generating vocabulary for en-pali Pali...")
            codes_file_pali = generate_vocab(pali_train_file, os.path.join(Config.OUT_DIR, "codes_file_san"))

            return (corpus_to_tensors(eng_train_file, codes_file_eng, Config.OUT_DIR, "en_pali_english_full"),
                    corpus_to_tensors(pali_train_file, codes_file_pali, Config.OUT_DIR, "en_pali_pali_full"))
            
def create_multilingual_dataset(eng_san_eng_train, eng_san_san_train, eng_pali_eng_train, eng_pali_pali_train):
    
    # print(len(eng_pali_eng_train))
    # print(len(eng_pali_pali_train))
    
    features = [torch.cat((torch.tensor([Config.SAN_TOKEN]), feature)) for feature in eng_san_eng_train] #add san token (10998)
    features = features[:75000]

    features.extend([torch.cat((torch.tensor([Config.PLI_TOKEN]), feature)) for feature in eng_pali_eng_train]) #add pali token (10997)    
    features = features[:27100]

    labels = eng_san_san_train
    labels = labels[:75000]
    
    labels.extend(eng_pali_pali_train)
    labels = labels[:27100]
    
    print(len(features))
    print(len(labels))
    
    return features, labels
