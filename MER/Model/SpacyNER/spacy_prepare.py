# Convert json file to spaCy format.
import logging
import argparse
import sys
import os
import json
import pickle

def main(input_file=None, output_file=None):
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
if __name__ == '__main__':
    for type in ["train","test"]:
        input_file = "../../data/ner_{}.json".format(type)
        output_file = "../../data/ner_prepare_{}.pkl".format(type)
        main(input_file, output_file)