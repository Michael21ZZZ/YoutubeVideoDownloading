import argparse
from Model.SpacyNER import train_model as spacy_train
from Model.SpacyNER import evaluation as spacy_eval
from Model.CRF import train_model as crf_train

from Model.LSTM_CRF import train_model as lstm_crf_train
import os
from pathlib import Path
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--model_type", required=True, type=str, choices=['lstm_crf','crf','spacy'])

    args = args.parse_args()
    if os.path.exists("saved_model") is False:
        Path("saved_model").mkdir(exist_ok=True)
    if os.path.exists("output") is False:
        Path("output").mkdir(exist_ok=True)


    if args.model_type == "spacy":
        spacy_train.model_train(n_iter=10)
        print(spacy_eval.evaluate())
    elif args.model_type == "crf":
        # already contain the parameter search in the main training function
        crf_train.model_train()
        crf_train.evaluate()
    else:
        lstm_crf_train.model_train()


    # spacy
    # lstm-crf
    # crf
