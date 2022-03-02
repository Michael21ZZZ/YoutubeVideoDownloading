import re
import pickle
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
#https://stackoverflow.com/questions/44827930/evaluation-in-a-spacy-ner-model

def spacy_annoate(raw_data, nlp):
    # raw data: list of description
    # annoate_data: panda datafrane contain the location of the entity
    # assert there is no intersection between entities
    docs = []
    for i in raw_data:
        docs.append(nlp(i))
    # docs = [nlp(i) for i in raw_data]
    raw_data_tokens = [list(i) for i in raw_data]
    begin_entity = " <MT> "
    end_entity = " </MT> "
    annoated_result = []
    entity_per_doc = []
    for doc, line in zip(docs, raw_data_tokens):
        line_entity = []
        if len(doc.ents) == 0:
            annoated_result.append("<SPACY NONE> " + "".join(line))
            entity_per_doc.append("<SPACY NONE>")
            continue
        for i, ent in enumerate(doc.ents):
            start_index = ent.start_char + i * 2
            end_index = ent.end_char + i * 2
            assert ent.text == "".join(line[start_index: end_index])
            # assert "".join(line[start_index: end_index]) == ent.text

            line.insert(start_index, begin_entity)
            line.insert(end_index + 1, end_entity)
            line_entity.append(ent.text)
        annoated_result.append("".join(line))
        entity_per_doc.append("|".join(line_entity))
    return annoated_result, entity_per_doc

def evaluate(ner_model="./saved_model/trained_spacy.model", src_file="./data/preprocess/spacy/spacy_prepare_test.pkl"):
    scorer = Scorer()
    ner_model = spacy.load(ner_model)
    src_data = pickle.load(open(src_file, 'rb'))
    for input_, annot in src_data:
        doc_gold_text = ner_model.make_doc(input_)
        annot = annot['entities']
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

def get_annoate(ner_model, check_file, save_file):
    data = pickle.load(open(check_file,'rb'))
    data = [i[0].strip() for i in data]
    annoated_result, entity_per_doc = spacy_annoate(data, ner_model)
    with open(save_file,'w') as f1:
        for line in annoated_result:
            f1.write(line + "\n")

def inference(check_file, save_file, model_path="./saved_model/trained_spacy.model"):
    ner_model = spacy.load(model_path)
    data = open(check_file,'r').readlines()

    data = [i.strip().replace("<MT>"," ").replace("</MT>"," ") for i in data]
    annoated_result, entity_per_doc = spacy_annoate(data, ner_model)
    with open(save_file,'w') as f1:
        for line in annoated_result:
            f1.write(line + "\n")

if __name__ == '__main__':
    # trained model
    # acc {'ENTITY': {'p': 75.85736554949337, 'r': 78.44045939955672, 'f': 77.1272907379891}}
    # raw model
    # 'ENTITY': {'p': 41.97930142302717, 'r': 26.15353616764054, 'f': 32.2284295468653}
    # trained model
    # {'p': 36.619036086219424, 'r': 23.721368057734544, 'f': 28.791773778920305}}

    model_path = "../saved_model/trained_spacy.model"
    ner_model = spacy.load(model_path)
    src_file = "../../data/preprocess/spacy/spacy_prepare_test.pkl"
    print(evaluate(ner_model, src_file))
    get_annoate(ner_model, src_file, "./spacy_result.txt")

