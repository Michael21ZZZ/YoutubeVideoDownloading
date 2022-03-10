import pandas as pd
import pickle
def reload_samples(file_name):
    data = pickle.load(open(file_name, "rb"))
    text = [" ".join([j[0] for j in i])for i in data]
    entities = [" ".join([j[1] for j in i]) for i in data]
    return text, entities

def reload_for_bert(file_name):
    data = pickle.load(open(file_name, "rb"))
    text = [[" ".join(j) + "\n" for j in i]for i in data]
    text = [i+["\n"] for i in text]
    return text

def main():
    bert=True
    file_name = "/home/local/ASUAD/yichuan1/Carry/data/sequence_{}_BLO.pkl"
    for type in ["train",'test']:
        file_name_type = file_name.format(type)
        save_tag_name = "./{}_MT.tags.txt".format(type)
        save_txt_name = "./{}_MT.words.txt".format(type)
        if bert:
            text = reload_for_bert(file_name_type)
            with open("/home/local/ASUAD/yichuan1/Carry/transformers/examples/ner/{}.txt.tmp".format(type), 'w') as f1:
                for lines in text:
                    for line in lines:
                        f1.write(line)
        else:
            text, tag = reload_samples(file_name_type)
            with open(save_tag_name, 'w') as f1:
                for line in tag:
                    f1.write(line.strip() + "\n")
            with open(save_txt_name, 'w') as f1:
                for line in text:
                    f1.write(line.strip() + "\n")
if __name__ == '__main__':
    main()


