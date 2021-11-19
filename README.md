
# Sanskrit Translation

Contains code for neural machine translation of Sanskrit into English.

## Dependencies

```
pip install subword_nmt datasets pdfminer pdfminer.six
```

### Tasks 
- Preprocessing refactoring
  - [x] Make logic for tokenization more general (i.e., can be applied to more kinds of corpora)
  - [ ] Clean up vocabulary generation logic - Should be able to create a vocab from a
        monolingual text file and use it anywhere
- Pretrain on Marati, Odiya, or Hindi
- Multilingual with Ancient Greek and Pali
- Expanding dataset size using new sources
  - Scraping internet 
- Using novel tokenization methods
  - Sanskrit embeddings https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/

### Timeline
#### Novemeber 22, 2021
- Transformer trained
- Pre-training language found, formatted, and preprocessed
- Refactor preprocesssing pipeline
