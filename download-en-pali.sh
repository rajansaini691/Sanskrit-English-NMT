#!/bin/bash

###########################################################################
# This script generates data/en-pali/train.en and data/en-pali/train.pli. #
# The other files in data/en-pali are intermediate and can be ignored. 	  #
###########################################################################

pip install indic_transliteration
mkdir -p data/
mkdir -p data/en-pali
cd data/en-pali
echo "Cloning raw data source"
git clone https://github.com/suttacentral/bilara-data
cd bilara-data/.scripts/bilara-io
./sheet_export.py vinaya vinaya.tsv --include root,translation+en
./sheet_export.py sutta sutta.tsv --include root,translation+en
cp vinaya.tsv ../../../
cp sutta.tsv ../../../
cd ../../../../..
echo "Cleaning and reformatting data..."
python Sanskrit-English-NMT/pali_tsv_to_corpora.py --corpus-root data/en-pali/ data/en-pali/sutta.tsv data/en-pali/vinaya.tsv
