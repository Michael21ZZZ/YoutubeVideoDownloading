"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1
import itertools
import pandas as pd
from sklearn_crfsuite import metrics
import os
import shutil

DATADIR = './data/preprocess/lstm_crf'
PARAMAS = {
    'dim': 300,
    # decrease more
    'dropout': 0.3,
    'num_oov_buckets': 1,
    'epochs': 50,
    'batch_size': 20,
    'buffer': 15000,
    'lstm_size': 80,
    'words': str(Path(DATADIR, 'vocab.words.txt')),
    'chars': str(Path(DATADIR, 'vocab.chars.txt')),
    'tags': str(Path(DATADIR, 'vocab.tags.txt')),
    'glove': str(Path(DATADIR, 'glove.npz')),
    'lr': 1e-3
}
# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    if isinstance(features, dict):
        features = features['words'], features['nwords']

    # Read vocabs and inputs
    dropout = params['dropout']
    words, nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.]*params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    embeddings = tf.nn.embedding_lookup(variable, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        # TODO: separately result
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params['lr']).minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


def fwords(name):
    return str(Path(DATADIR + "/description/", '{}.words.txt'.format(name)))


def ftags(name):
    return str(Path(DATADIR + "/description/", '{}.tags.txt'.format(name)))

def model_train():
    # Params
    params = PARAMAS
        # set up the logging file
    if os.path.exists('saved_model/lstm_crf'):
        shutil.rmtree('saved_model/lstm_crf')
    Path('saved_model/lstm_crf').mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('saved_model/lstm_crf/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers


    with Path('saved_model/lstm_crf/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)



    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train_MT'), ftags('train_MT'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('test_MT'), ftags('test_MT'))

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=5,
        inter_op_parallelism_threads=5)
    cfg = tf.estimator.RunConfig(session_config=session_conf,save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, 'saved_model/lstm_crf/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        if os.path.exists("./output") is False:
            Path('./output').mkdir(parents=True, exist_ok=True)
        with Path('output/lstm_test_result.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    # test acc = 0.76826996, f1 = 0.72103536, global_step = 2125, loss = 6.917531, precision = 0.69827586, recall = 0.7453284
    for name in ['test_MT']:
        write_predictions(name)

    #
    print("#"*20)
    print("Seperate Evaluation")
    seperate_eval('output/lstm_test_result.txt')

def predict_input_fn(line):
    # Words
    words = [w.encode() for w in line.strip().split()]
    nwords = len(words)

    # Wrapping in Tensors
    words = tf.constant([words], dtype=tf.string)
    nwords = tf.constant([nwords], dtype=tf.int32)

    return (words, nwords), None


def inference(inference_file, output_file, tag_filename, cnt_filename, modeldir='./saved_model/lstm_crf/model'):

    params = PARAMAS
    params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    params['glove'] = str(Path(DATADIR, 'glove.npz'))

    estimator = tf.estimator.Estimator(model_fn, modeldir, params=params)
    inference_list = open(inference_file,'r').readlines()
    inference_list = [i.replace("<MT>"," ").replace("</MT>"," ") for i in inference_list]
    name = inference_file.split("/")[-1]
    with open(fwords(name),'w') as f1:
        for line in inference_list:
            f1.write(line)

    ftag_file = ftags(name)
    with open(fwords(name), 'r') as f1:
        words = f1.readlines()
    with open(ftag_file, 'w') as f1:
        for line in words:
            line_tags = " ".join(["<Pad>"] * len(line.strip().split()))
            f1.write(line_tags + "\n")

    def write_predictions(name):
        # set up the fake tags
        Path('./saved_model/lstm_crf/score').mkdir(parents=True, exist_ok=True)
        with Path(tag_filename).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    write_predictions(name)
    read_result(output_file, tag_filename, cnt_filename, name)
    # test acc = 0.76826996, f1 = 0.72103536, global_step = 2125, loss = 6.917531, precision = 0.69827586, recall = 0.7453284

def read_result(output_file, tag_filename, cnt_filename, name):
    data = open(tag_filename).readlines()
    empty_index = [index for index, i in enumerate(data) if i == "\n"]
    empty_index = [-1] + empty_index
    data_list = []
    for i in range(len(empty_index) - 1):
        b = empty_index[i] + 1
        e = empty_index[i+1]
        data_list.append([i.split() for i in data[b:e]])

    stored_list = []
    mer_cnt = 0
    for ele in data_list:
        ele = list(filter(lambda x: len(x) == 3, ele))
        length_limit = len(ele)
        list_tokens = []
        for index, line in enumerate(ele):
            tokens = line[0]
            pred = line[-1]
            if pred == "B-MT":
                mer_cnt += 1
                tokens = "<MT> " + tokens
                if (index < length_limit - 1 and ele[index + 1][-1] != "I-MT") or (index == length_limit - 1):
                    tokens += " </MT>"
            if pred == "I-MT":
                if (index < length_limit - 1 and ele[index + 1][-1] != "I-MT") or (index == length_limit - 1):
                    tokens += " </MT>"

            list_tokens.append(tokens)
        stored_list.append(" ".join(list_tokens).replace("\n", ""))
    with open(output_file,'w') as f1:
        for line in stored_list:
            f1.write(line + "\n")

    with open(cnt_filename, 'a') as f2:
        f2.write(name + " " + str(mer_cnt) + "\n")

def seperate_eval(file_path):
    data = pd.read_csv(file_path, sep=" ", header=None)
    data = data.values.tolist()
    data_new = []
    line = []
    for i in data:
        if i[0] == "." and i[1] == "O":
            data_new.append(line.copy())
            line = []
        else:
            line.append(i)

    y_test = [[j[1] for j in i] for i in data_new]
    y_pre = [[j[2] for j in i] for i in data_new]
    labels = ["B-MT", "I-MT", "O"]
    metrics.flat_f1_score(y_test, y_pre,
                          average='weighted', labels=labels)
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    print(metrics.flat_classification_report(
        y_test, y_pre, labels=sorted_labels, digits=3
    ))