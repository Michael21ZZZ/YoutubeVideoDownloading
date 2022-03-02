import pandas as pd
import numpy as np
import json
import logging
import sys
from sklearn.model_selection import train_test_split
import re
from itertools import chain
import pickle
import os
import argparse
from collections import Counter
from pathlib import Path

def read_file(file_name, spacy_mode=True):
    data = open(file_name,'r').readlines()
    data = [i.strip() for i in data]
    data = [i.strip().rstrip() for i in data if (len(i) > 1)]
    print("The line of dataset is {}".format(len(data)))
    # train test split
    train = data[800:]
    test = data[:800]
    if spacy_mode:
        prefix = "spacy"
    else:
        prefix = "nonspacy"

    # train, test = train_test_split(data, random_state=123, test_size=0.25)


    for data, name in zip([train, test], ["train","test"]):

        with open("./data/preprocess/ner_{}.txt".format(name),'w') as f1:
            for line in data:
                f1.write(line.strip() + "\n")

        result = []
        list_sequence = []
        for line in data:
            flip = False
            token_list = line.split()
            len1 = len(result)
            for index, tokens in enumerate(token_list):
                if tokens == "<MT>":
                    flip = True
                elif "<MT>" in tokens:
                    flip = True
                    result.append((tokens.replace("<MT>","").replace("</MT>",""), "B-MT"))
                elif tokens == "</MT>":
                    try:
                        if result[-2][1] == "B-MT" and result[-1][1] == "B-MT":
                            result[-1][1] = "I-MT"
                        else:
                            th = 1
                            print("ERROR")
                        flip = False
                    except:
                        flip = False

                elif "</MT>" in tokens:
                    # "</MT>others"
                    if tokens[:len("</MT>")] == "</MT>":
                        result.append((tokens.replace("</MT>",""), "O"))
                    else:
                        if result[-1][1] == "B-MT":
                            result.append((tokens.replace("</MT>",""), "I-MT"))
                        else:
                            result.append((tokens.replace("</MT>", ""), "B-MT"))
                    flip = False
                elif flip:
                    if token_list[index-1] == "<MT>":
                        result.append((tokens, "B-MT"))
                    elif result[-1][1] == "B-MT" or result[-1][1] == "I-MT":
                        result.append((tokens, "I-MT"))
                else:
                    result.append((tokens, "O"))

            result.append((".","O"))
            list_sequence.append(result[len1:])
        if spacy_mode:
            result = [[i[0], i[1] if i[1] == "O" else "ENTITY"] for i in result]
            list_sequence = [[[j[0], j[1] if j[1] == "O" else "ENTITY"] for j in i] for i in list_sequence]

        else:

            # TODO: use the NER model from https://github.com/guillaumegenthial/tf_ner
            pass
        result_df = pd.DataFrame(result, columns=["Word","Tag"])
        result_df.to_csv("./data/preprocess/{}_{}_data.tsv".format(prefix, name),index=None, sep="\t")
        with open("./data/preprocess/{}_{}_BLO.pkl".format(prefix, name), 'wb') as f1:
            pickle.dump(list_sequence, f1)
    return prefix

def tsv_to_json_format(input_path, output_path, unknown_label):
    try:
        f = open(input_path, 'r')  # input file
        fp = open(output_path, 'w')  # output file
        data_dict = {}
        annotations = []
        label_dict = {}
        s = ''
        start = 0
        f.__next__()
        for line in f:
            if line[0:len(line) - 1] != '.\tO':
                word, entity = line.split('\t')
                s += word + " "
                entity = entity[:len(entity) - 1]
                if entity != unknown_label:
                    if len(entity) != 1:
                        d = {}
                        d['text'] = word
                        d['start'] = start
                        d['end'] = start + len(word) - 1
                        try:
                            label_dict[entity].append(d)
                        except:
                            label_dict[entity] = []
                            label_dict[entity].append(d)
                start += len(word) + 1
            else:
                data_dict['content'] = s
                s = ''
                label_list = []
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if (label_dict[ents][i]['text'] != ''):
                            l = [ents, label_dict[ents][i]]
                            for j in range(i + 1, len(label_dict[ents])):
                                if (label_dict[ents][i]['text'] == label_dict[ents][j]['text']):
                                    di = {}
                                    di['start'] = label_dict[ents][j]['start']
                                    di['end'] = label_dict[ents][j]['end']
                                    di['text'] = label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text'] = ''
                            label_list.append(l)

                for entities in label_list:
                    label = {}
                    label['label'] = [entities[0]]
                    label['points'] = entities[1:]
                    annotations.append(label)
                data_dict['annotation'] = annotations
                annotations = []
                json.dump(data_dict, fp)
                fp.write('\n')
                data_dict = {}
                start = 0
                label_dict = {}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None

def pickle_to_txt(file_name, output_name):
    data = pickle.load(open(file_name,'rb'))
    with open(output_name,'w') as f1:
        for lines in data:
            for line in lines:
                f1.write(" ".join(line).replace("\n","") + "\n")
            f1.write("\n")

def spacy_polish(input_file=None, output_file=None):
    try:
        training_data = []
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1, label))


            training_data.append((text, {"entities" : entities}))

        print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None

def crf_polish(file_path, save_path):
    def prepare_data(token_list):
        text_token = [i.split("\t") for i in token_list]
        text_token = [(i[0].strip(), i[1].strip()) for i in text_token]
        return text_token
    data = open(file_path, 'r').readlines()
    new_line_index = [-1] + [idx for idx, i in enumerate(data) if i.strip() == ".\tO"]
    output = []
    for idx in range(len(new_line_index) - 1):
        start = new_line_index[idx] + 1
        end = new_line_index[idx + 1]
        th = prepare_data(data[start:end])
        output.append(th)
    pickle.dump(output, open(save_path, 'wb'))

def lstm_crf_polish(input_file):


    ####################################################################
    # A. covert the file into tags and words
    for type in ["train",'test']:
        def reload_samples(file_name):
            data = pickle.load(open(file_name, "rb"))
            text = [" ".join([j[0] for j in i]) for i in data]
            entities = [" ".join([j[1] for j in i]) for i in data]
            return text, entities
        file_name_type = input_file.format(type)
        save_tag_name = "./data/preprocess/lstm_crf/{}_MT.tags.txt".format(type)
        save_txt_name = "./data/preprocess/lstm_crf/{}_MT.words.txt".format(type)

        text, tag = reload_samples(file_name_type)
        with open(save_tag_name, 'w') as f1:
            for line in tag:
                f1.write(line.strip() + "\n")
        with open(save_txt_name, 'w') as f1:
            for line in text:
                f1.write(line.strip() + "\n")

    ####################################################################
    # B. set up the vocab
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return '{}.words.txt'.format(name)

    os.chdir("./data/preprocess/lstm_crf")
    print('Build vocab words (may take a while)')
    counter_words = Counter()
    # for n in ['train', 'testa', 'testb']:
    for n in ['train_MT', 'test_MT']:
        with Path(words(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= 1}

    with Path('vocab.words.txt').open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words

    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path('vocab.chars.txt').open('w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))

    print('- done. Found {} chars'.format(len(vocab_chars)))

    # 3. Tags
    # Get all tags from the training set

    def tags(name):
        return '{}.tags.txt'.format(name)

    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    with Path(tags('train_MT')).open() as f:
        for line in f:
            vocab_tags.update(line.strip().split())
    with Path(tags('test_MT')).open() as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with Path('vocab.tags.txt').open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))

    ####################################################################
    # C. read word embedding
    with Path('vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    print('./data/lstm_crf/Reading GloVe file (may take a while)')
    with Path('glove.840B.300d.txt').open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed('glove.npz', embeddings=embeddings)

def main(args):

    # train/test split
    if args.preprocess_type == "spacy":
        prefix = read_file(args.input_file, spacy_mode=True)

    else:
        prefix = read_file(args.input_file, spacy_mode=False)

    if os.path.exists("./data/preprocess/spacy") is False:
        Path("./data/preprocess/spacy").mkdir(exist_ok=True)
    if os.path.exists("./data/preprocess/lstm_crf") is False:
        Path("./data/preprocess/lstm_crf").mkdir(exist_ok=True)
    if os.path.exists("./data/preprocess/crf") is False:
        Path("./data/preprocess/spacy").mkdir(exist_ok=True)
    for type in ['train','test']:

        tsv_to_json_format("./data/preprocess/{}_{}_data.tsv".format(prefix, type),'./data/preprocess/{}_{}.json'.format(prefix, type), 'abc')
        pickle_to_txt("./data/preprocess/{}_{}_BLO.pkl".format(prefix, type), "./data/preprocess/{}_{}.txt".format(prefix, type))

        # specifc for spacy
        if args.preprocess_type == "spacy":
            input_file = "./data/preprocess/{}_{}.json".format(prefix, type)
            output_file = "./data/preprocess/spacy/spacy_prepare_{}.pkl".format(type)
            spacy_polish(input_file, output_file)
        elif args.preprocess_type == "crf":
            input_file = "./data/preprocess/{}_{}_data.tsv".format(prefix, type)
            output_file = "./data/preprocess/crf/crf_prepare_{}.pkl".format(type)
            crf_polish(input_file, output_file)
        else:
            input_file = "data/preprocess/"+prefix+"_{}_BLO.pkl".format(type)
            lstm_crf_polish(input_file)
            break

if __name__ == '__main__':
    # TODO: agg all the preprocess together
    arg = argparse.ArgumentParser()
    arg.add_argument("--input_file", help="Input File Name for preprocess", default="", required=True)
    arg.add_argument("--preprocess_type", help="The preprocess mode from ['spacy', 'lstm_crf','crf'] ", required=True,
                     choices=['spacy', 'lstm_crf','crf'])
    arg = arg.parse_args()
    main(arg)
    # read_file(arg.file_name, spacy_mode=arg.is_spacy)
    # tsv_to_json_format("../data/prepare_train_data.tsv", '../data/ner_train.json', 'abc')
    # tsv_to_json_format("../data/prepare_test_data.tsv", '../data/ner_test.json', 'abc')
    # # pickle_to_txt("../data/sequence_train_BLO.pkl","../data/train.txt")
    # # pickle_to_txt("../data/sequence_test_BLO.pkl","../data/test.txt")
