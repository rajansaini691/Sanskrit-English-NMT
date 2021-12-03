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
    sentence = sentence.reshape([max_sequence_length, 1])

    src_pad_mask = model.make_len_mask(sentence)
    
    src = model.encoder(sentence)
    src = model.pos_encoder(src)
    
    encoded = model.transformer.encoder(src.to(Config.DEVICE))
    

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
                
                trg_pad_mask = model.make_len_mask(candidates)
                
                trg = model.decoder(candidates)
                trg = model.pos_decoder(trg)
                
                # print('trg', trg.shape)
                output = model.transformer.decoder(trg.to(Config.DEVICE),
                                                   encoded, 
                                                   memory_key_padding_mask=src_pad_mask,
                                                   tgt_key_padding_mask=trg_pad_mask)
                
                output = model.fc_out(output)
                
                # output = output.transpose(0, 1).transpose(1, 2)
                # print(output)
                predicted_id = torch.nn.functional.softmax(output, dim=-1)
                # predicted_id = output.argmax(2)[-1].item()
                # print(predicted_id.shape)
                # print(predicted_id)
                softmaxes = predicted_id[:, -1].to('cpu')[-1]
                # print(softmaxes.shape)
                # print(softmaxes)
           
                indicies = np.argpartition(softmaxes.to('cpu'), (k*-1))[(k*-1):]
                # print(indicies)
                #add potential new candidates to priority queue
                for index in indicies:
                    sm_score = softmaxes[index]

                    new_candidate = np.append(candidates, [[index]], axis=0)
                    new_score = np.multiply(score, sm_score)
                    
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
        for index in range(len(eng_train[:10])):
            
            best_score = beamSearch(eng_train[index], 3, model)
            # memory = model.transformer.encoder(model.pos_encoder(model.encoder(eng_test[index])))

            # out_indexes = [1, ]

            # for i in range(50):
            #     trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(Config.DEVICE)

            #     output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
            #     out_token = output.argmax(2)[-1].item()
            #     out_indexes.append(out_token)
            #     if out_token == 0:
            #         break
            # print(out_indexes)
            translation = convertToText(best_score)            
            translations.append("".join(translation[1:]))
            
            # bleu = metrics.BLEU()
            # res = bleu.sentence_score(translations[index], [references[index]])
            # print(f"BLEU score: ", res)
            
            if index % 10 == 0:
                print(f'[Testing] at {index}')

    print('trans', translations)
    print('refs', references[:10])
    bleu = metrics.BLEU()
    res = bleu.corpus_score(translations, [references])
    print(f"BLEU score: ", res)
