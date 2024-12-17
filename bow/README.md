# BOW
Bag Of Words

## Overview
train / use bag of words to predict the labels of sentences, in input data, sentence col must be named 'SENTENCE' and label 'LABEL'

## Usage
* train
  python3 bow.py -i /tmp/bow_input.csv -o /tmp/bow -a train

* prediction
  python3 bow.py -i "j'aime les chamalots" -o /tmp/bow -a predict
  OR
  python3 bow.py -i /tmp/bow_predict.csv -o /tmp/bow -a predict

