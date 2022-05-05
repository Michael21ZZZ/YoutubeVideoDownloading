import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# data = pd.read_csv("train_MT.words.txt", sep = ' ', header=None).fillna(method="ffill")
# data.tail(10)

with open("train_MT.words.txt", 'r') as train:
	train_data = train.readlines()
lines_all = []
for data in train_data:
	lines = data.split()
	lines_all.append(lines)
print(len(lines_all))

with open("train_MT.tags.txt", 'r') as tags:
	tag_data = tags.readlines()
tags_all = []
for tag in tag_data:
	tags = tag.split()
	tags_all.append(tags)
print(len(tags_all))

# print(lines_all[4172])
# print(tags_all[4172])

zipped_lists = zip(lines_all, tags_all)

# data_df = pd.DataFrame(columns = ["Word", "Tag"])
dfs = []
for line_list, tag_list in zipped_lists:
	# print('a')
	# print(len(line_list))
	# print(len(tag_list))
	temp_df = pd.DataFrame(list(zip(line_list, tag_list)), columns = ["Word", "Tag"])
	# print(temp_df.head())
	dfs.append(temp_df)
	# print(len(line_list))
	# print(len(tag_list))
all_data = pd.concat(dfs)
print(all_data)

MAX_LEN = 75
bs = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels