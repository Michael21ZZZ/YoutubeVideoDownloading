
#from Model.SpacyNER import evaluation as spacy
#from Model.CRF import train_model as crf
from Model.LSTM_CRF import train_model as lstm_crf
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--model_type", required=True, choices=["lstm_crf",'crf', 'spacy'])
    args.add_argument("--input_file", required=True, type=str)
    args.add_argument("--output_file", required=True, type=str)

    args = args.parse_args()
    if args.model_type == "lstm_crf":
        lstm_crf.inference(args.input_file, args.output_file)
    elif args.model_type == "crf":
        crf.inference(args.input_file, args.output_file)
    else:
        spacy.inference(args.input_file, args.output_file)
