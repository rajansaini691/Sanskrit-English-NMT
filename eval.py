import os
import torch
import math
import json
import argparse
import model
import numpy as np
from datasets import load_dataset
from model import Transformer
from preprocess import convert_itihasa_dataset_to_tensors, create_token_dict, load_marathi_dataset
from config import Config
import sacrebleu.metrics as metrics
import sacrebleu

parser = argparse.ArgumentParser(description='BPE tokenization.')
parser.add_argument('-en_dict')
parser.add_argument('-ha_dict')
parser.add_argument('-ref')
parser.add_argument('-codes')
parser.add_argument('-eval', default='text')
parser.add_argument('-model')
args = parser.parse_args()

# beam search algorithm
def beamSearch(sentence, k, model):
    """
    Beam search algorithm, sentence is input sentence, k is amount of top k, model is model
    """
    max_sequence_length = sentence.shape[0]

    best_scores = []
    best_scores.append((1, np.ones((1, 1))))
    
    # encode to get encoder output
    src = model.encoder(sentence)
    src = model.pos_encoder(src)
    
    # encoder_input = torch.reshape(sentence, (1, -1))
    encoded = model.transformer.encoder(src.to(
        Config.DEVICE))
    

    for i in range(1, max_sequence_length):
        new_seqs = PriorityQueue(k)
        for score, candidate in best_scores:
            
            if candidate[-1] == 0: #if EOS token reached, add to priority queue
                if not new_seqs.full():
                    new_seqs.put((score, list(candidate)))
                else:
                    if new_seqs.queue[0][0] < score:
                        new_seqs.get()  # pop the one with lowest score
                        new_seqs.put((score, list(candidate)))

            else: #otherwise decoder next token
                candidates = torch.from_numpy(
                    np.array(candidate, dtype=int))
                
                trg = model.decoder(candidates)
                trg = model.pos_decoder(trg)

                # print('trg', trg.shape)
                output = model.transformer.decoder(trg.to(Config.DEVICE),
                                                   encoded)
                # print(output)
                predicted_id = torch.nn.functional.log_softmax(output, dim=-1)
                softmaxes = predicted_id[-1].to('cpu')[-1]
                indicies = np.argpartition(softmaxes.to('cpu'), (k*-1))[(k*-1):]

                #add potential new candidates to priority queue
                for index in indicies:
                    sm_score = softmaxes[index]

                    new_candidate = np.append(candidates, [[index]], axis=0)
                    new_score = np.add(score, sm_score)
                    
                    if not new_seqs.full():
                        new_seqs.put((new_score, list(new_candidate)))
                    else:
                        if new_seqs.queue[0][0] < new_score:
                            new_seqs.get()  # pop the one with lowest score
                            new_seqs.put((new_score, list(new_candidate)))

            #append to new best_scores
            best_scores = []
            while not new_seqs.empty():
                best_scores.append(new_seqs.get())

    #get overall best score
    best_score = -1 * math.inf
    cand = []
    for score, candidate in best_scores:
        if score > best_score:
            best_score = score
            cand = candidate

    return cand


def convertToText(sequence):
    '''
    Convert indexes to text
    '''
    codes_file_san = open(os.path.join(Config.OUT_DIR, 'codes_file_san'))
    san_dict = create_token_dict(codes_file=codes_file_san, dtype='target')

    with open('san_dict.json', 'w') as fp:
        json.dump(san_dict, fp)
        
    translation = []
    key_list = list(san_dict.keys())
    val_list = list(san_dict.values())
    
    for byte in sequence:
        if byte == Config.PADDING_IDX:
            return translation
        position = val_list.index(byte)
        
        word = key_list[position]

        if word[-4:] == '</w>': 
            translation.append(word[:-4] + " ")
        else:
            translation.append(word)
                
    return translation


if __name__ == "__main__":
    from queue import PriorityQueue
    
    model_path = args.model

    # Initialize out dir
    if not os.path.exists(Config.OUT_DIR):
        os.mkdir(Config.OUT_DIR)

    print('Device:', Config.DEVICE)

    dataset = load_dataset("rahular/itihasa")

    training_data = dataset['train']
    validation_data = dataset['validation']
    test_data = dataset['test']
    
    # eng_train, mar_train=load_marathi_dataset(os.path.join(Config.DATA_DIR, "en-mr"))

    eng_train, san_train, eng_val, san_val, eng_test, san_test = convert_itihasa_dataset_to_tensors(training_data, validation_data, test_data)
    # load_marathi_dataset(os.path.join(Config.DATA_DIR, "en-mr"))

    model = Transformer(
        Config.SRC_VOCAB_SIZE,
        Config.TRG_VOCAB_SIZE,
        Config.HIDDEN_SIZE, 
        Config.NUM_LAYERS,
    ).to(Config.DEVICE)
            
    if Config.LOAD_MODEL:
        checkpoint = torch.load(model_path,
                                map_location=Config.DEVICE)

        start_batch = checkpoint["batch"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])

    model.load_state_dict(checkpoint["model_state"])
    model.eval()    
    
    #do beam search each each input
    translations = []
    
    bleu_score = 0
    
    references = []
    for reference in test_data:
        if reference['translation']['sn'] != '':
            references.append(reference['translation']['sn'])

    with torch.no_grad():
        for index in range(len(eng_test[:10])):
            best_score = beamSearch(eng_test[index], 3, model)
            translation = convertToText(best_score)            
            translations.append("".join(translation[1:]))
            
            bleu = metrics.BLEU()
            res = bleu.sentence_score(translations[index], [references[index]])
            print(f"BLEU score: ", res)
            
            if index % 10 == 0:
                print(f'[Testing] at {index}')

    print('trans', translations)
    print('refs', references[:10])
    bleu = metrics.BLEU()
    res = bleu.corpus_score(translations, [references])
    print(f"BLEU score: ", res)
