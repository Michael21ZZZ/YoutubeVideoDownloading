# Medical NER 

## Methods:
- [Spacy](https://allenai.github.io/scispacy/)
- [LSTM_CRF](https://github.com/guillaumegenthial/tf_ner)
- [CRF](https://sklearn-crfsuite.readthedocs.io/en/latest/)

## Install requirements
This program is only tested on python3.6. 
  
    pip install -r requirements.txt


## Data Preprocess:
Different methods need different pre-process steps. For example, for the method **Spacy**, you should run:
    
    
    export INPUT_FILE=./data/clean_annotation.txt
    python data_preprocess.py --input_file $INPUT_FILE --preprocess_type spacy
   
You can try **"lstm_crf", "crf" and "spacy"** options for different models. 

Noticed that, **lstm_crf** need the pre-trained language model. So you should download it firstly, then unzip the file in **./data/preprocess/lstm_crf** directory and finally you can run previous pre-process script. 

    wget  http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip -P ./data/preprocess/lstm_crf
    unzip ./data/preprocess/lstm_crf/glove.840B.300d.zip -d  ./data/preprocess/lstm_crf
    

## Model Training:
You can easily train Spacy model by running:
    
    python train_model.py --model_type spacy

For other two models, you just need to change the model name to **"lstm_crf"** or **"crf"**. 


## Inference:
When you have done the model training, you can annotate a new file by:
       
    export INPUT_FILE=./data/clean_annotation.txt
    export OUTPUT_FILE=./data/output_test.txt 
    python Inference.py --model_type spacy --input_file $INPUT_FILE --output_file $OUTPUT_FILE 

You can check the annoated text in **./data/output_test.txt**. Also for other models, you need to change the model_type to **"lstm_crf"** or **"crf"**

# TODO: Documentation
1.  Function introduction.
2.  Reorganize the folder.
3.  Three walk through tutorials. 
4.  **Evaluation by class.** 
5.  Add requirements.txt.
