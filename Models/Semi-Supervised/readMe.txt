DATA:
labeled_all_features.csv: This is the labeled dataset. We are using a total of 22 features and 566 datapoints (or videos)
unlabeled_all_features.csv: This is the unlabeled dataset. We were able to gather a total of 24 features but we are only using
22 features to test our model on. These are the same set of 22 features that are present in the labeled dataset. This is the clean
dataset which has a total of 256 datapoints after cleaning the unlabeled set.

Model Scripts:
co-training-sklearn-only.py:

This file reads in the two data files unlabeled_all_features.csv and labeled_all_features.csv.
After some pre-processing, it produces classification reports for 5 different classifiers, namely:

- Random Forest Classifier (max_depth = 15, n_estimators = 200)
- Extra Trees Classifier
- Histogram-based Gradient Boosting Classifier
- AdaBoostClassifier
- Gradient Boosting Classifier

These classification reports show the average F1-score, Precision, Recall and Accuracy for the different algorithms



