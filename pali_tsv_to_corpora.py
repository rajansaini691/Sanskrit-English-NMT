"""
Converts the Pali tsv's generated by bilara-io into separate English and
Pali corpora

Example usage:
    ./pali_tsv_to_corpora.py vinaya.tsv sutta.tsv ...

    Outputs:
        corpus.en
        corpus.pali

    Line n of corpus.en should correspond to line n of corpus.pali
"""
import argparse
import pandas as pd
import numpy as np
import os
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root")
    parser.add_argument("pali_tsv_list", nargs="+")
    args = parser.parse_args()

    english_corpus = ""         # Something like "en_sentence_1\n en_sentence2\n..."
    pali_corpus = ""            # Something like "pali_sentence_1\n pali_sentence2\n..."

    for pali_tsv in args.pali_tsv_list:
        df = pd.read_csv(pali_tsv, sep='\t', header=0, low_memory=False)

        # Filter out rows with nonexistent Pali
        nonexistent_pali_mask = df['root-pli-ms'].apply(lambda x: isinstance(x, str))
        df = df[nonexistent_pali_mask]

        # Remove rows that contain no letters (i.e. rows with just numbers and dashes)
        numbers_and_dashes = set(str(i) for i in range(10))
        numbers_and_dashes.add('-')
        numbers_and_dashes.add('_')
        numbers_and_dashes.add(' ')
        no_letters_mask = df['root-pli-ms'].apply(lambda x: not set(x).issubset(numbers_and_dashes))
        df = df[no_letters_mask]

        # Merge the translations and remove rows with nonexistent translations
        en_trans_columns = [column_name for column_name in df.columns.values if 'translation-en' in column_name]
        merged_translations = df[en_trans_columns].apply(
                lambda translation: [x for x in translation if isinstance(x, str) and len(x.strip()) > 0], axis=1)
        df = df.assign(merged_translations=merged_translations)
        no_trans_mask = merged_translations.apply(lambda trans_list: len(trans_list) > 0)
        df = df[no_trans_mask]
        merged_translations_flattened = df['merged_translations'].apply(
                lambda translations: translations[0])
        df = df.assign(merged_translations=merged_translations_flattened)
        
        # Create the corpora
        pali_corpus += df['root-pli-ms'].to_csv(index=False, header=False)
        english_corpus += df['merged_translations'].to_csv(index=False, header=False)

    # Transliterate the pali corpus from roman to devanagari
    pali_corpus = transliterate(pali_corpus, sanscript.IAST, sanscript.DEVANAGARI)

    # Write the pali corpora
    with open(os.path.join(args.corpus_root, 'train.en'), 'w') as f:
        f.write(english_corpus)
    with open(os.path.join(args.corpus_root, 'train.pli'), 'w') as f:
        f.write(pali_corpus)
