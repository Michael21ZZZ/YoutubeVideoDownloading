import pickle
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import os
import subprocess
# New entity labels
# Specify the new entity labels which you want to add here
# LABEL = [ 'B-MT', 'O']
LABEL = ['ENTITY', 'O']

# Loading training data

print(os.listdir("./data/preprocess"))

with open('./data/preprocess/spacy/spacy_prepare_train.pkl', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

def model_train(model="en_core_sci_sm", model_path="./saved_model/trained_spacy.model",new_model_name="spacy_train", n_iter=1000):
    """Setting up the pipeline and entity recognizer, and training the new entity."""
    if model is not None:

        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    for i in LABEL:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    best_losses = 1e8
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)
            if itn % 100 == 0 and losses['ner'] < best_losses:
                best_losses = losses['ner']
                save_model(model_path, new_model_name,nlp, test_text ='Gianni Infantino is the president of FIFA.')



    # Test the trained model
    test_text = 'Gianni Infantino is the president of FIFA.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # Save model

def save_model(output_dir, new_model_name, nlp, test_text):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    model = "en_core_sci_sm"
    new_model_name = 'new_model'
    output_dir = "../saved_model/trained_spacy.model"
    n_iter = 1000
    main(model, output_dir, n_iter)