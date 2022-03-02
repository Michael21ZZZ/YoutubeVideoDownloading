## 1. Preprocess the data
    
            
        python spacy_prepare.py

Attention: Please run **data_preprocess.py** before this step. 

## 2. Train the model

    
        python model_train.py
        
 
This training is based on ``en_core_sci_sm" model and it save the trained model under **saved_model** directory.

## 3. Test the model 

    
        python evaluation.py
        

The evaluation result will show up in the standard output. The test result is stored in **spacy_result.txt** under this directory


