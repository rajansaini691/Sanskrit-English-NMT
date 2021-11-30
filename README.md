
# Sanskrit Translation

Contains code for neural machine translation of Sanskrit into English.

## Dependencies

### Conda
Setup conda environment:
```
conda create -n py39 python=3.9 anaconda
```

Install pytorch:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Install other dependencies
```
conda install tensorboard pandas
```

### Pip
Next, you need to install the following dependencies using `pip`.
```
pip install subword_nmt datasets pdfminer pdfminer.six indic_transliteration
```

### Tasks 
- Preprocessing refactoring
  - [x] Make logic for tokenization more general (i.e., can be applied to more kinds of corpora)
  - [x] Clean up vocabulary generation logic - Should be able to create a vocab from a
        monolingual text file and use it anywhere
- Pretrain on Marati, Odiya, or Hindi
- Multilingual with Ancient Greek and Pali
- Expanding dataset size using new sources
  - Scraping internet 
- Using novel tokenization methods
  - Sanskrit embeddings https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

### Timeline
#### November 22, 2021
- Transformer trained
- Pre-training language found, formatted, and preprocessed
- Refactor preprocesssing pipeline

#### November 29, 2021
- Have 1-2 pretrained models on pretraining data (Marathi)
- Find multilingual dataset
  - Ancient Greek
  - Pali
  - Latin (maybe) 
- Expanding dataset size using new sources

#### Final Week
- Fine-tune multiple pretrained models 
  - Different multiliguial combinations, and pure Sankrit
- Evaluation/Inference (sacreBLEU)
- Report/Presentation
