import os
import glob
from Model.LSTM_CRF import train_model as lstm_crf
import argparse

def get_processed_filename_set(output_dir):
    processed_filename_set = os.listdir(output_dir)
    processed_filename_set = [i.split('.')[0] for i in processed_filename_set]
    return processed_filename_set


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", required=True, type=str)
    args.add_argument("--output_dir", required=True, type=str)
    args.add_argument("--tag_dir", required=True, type=str)
    args.add_argument("--cnt_file", required=True, type=str)

    args = args.parse_args()

    processed_filename_set = get_processed_filename_set(args.output_dir)
    processed=0
    
    filenames = glob.glob(args.input_dir + "*.txt")
    print("remaining : ", len(filenames) - len(processed_filename_set))

    escape_list = ["YiQfe-oz2C4", "Q6OtzmR3eXo", "eqmi2ns7CKo"]
    escape_list = ["94jOCXHwlvw", "PpuiO6WJxic", "ZBFs0cbadqo", "F1YDR2S7SPU", "jxbbBmbvu7I",
                    "ZAjVQWbMlE", "6L7ZGr8qji0", "bSL6ax3qxaU", "urW-SmYnYHM", "OyCHKD2o_Yk",
                    "05ZltS-7Mto", "uOIS1vA1FS8", "uOIS1vA1FS8", "0f9ZK-1MkoM", "YA-YOXz9n98",
                    "y6bmCDjR4dI", "d4lepb2t4nE", "oIdrn7hLbGk", "8BhrVaOMUlQ", "7gir5BAOmGk",
                    "LOa-uXkn4Ds", "Bz5AjMAA8fg", "1OZyfo4NdLA", "rJiD9C3k__c", "toffLT1EQUw"]
    for filename in filenames:
        pre_name = filename.split('.')[0].split('/')[-1]
        if pre_name in escape_list:
            continue
        if pre_name not in processed_filename_set:
            output_file = args.output_dir + pre_name + ".txt"
            tag_file = args.tag_dir + pre_name + ".tags.txt"
            cnt_file = args.cnt_file
            with open("./text_output/content/checkfile.txt", 'a') as f:
                f.write(filename + "\n")
            lstm_crf.inference(filename, output_file, tag_file, cnt_file)
            processed += 1
            print('already processed %d'%processed)


