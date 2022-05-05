This directory has 3 files.

labeled_all_features.csv: This file contains the 
labeled data features which is read in by the
model script.

unlabeled_video_data.csv: This file contains the 
unlabeled data. This is read in the by model
script and some data pre-processing is done before
training different models on it.

classifier.py (the model script): This script
trains both supervised and un-supervised models
on the data (labeled_all_features.csv) and outputs
the results for each model. It also uses the 
unlabeled data and predicts the videos with
the highest predicted probability and outputs them
in the console. 