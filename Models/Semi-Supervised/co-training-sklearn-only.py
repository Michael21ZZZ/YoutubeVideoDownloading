# Importing important ML libraries / functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.semi_supervised import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

# Data Pre-processing and other libraries
import pandas as pd
import numpy as np
import warnings

# Adding this to not display warnings. Makes the output cleaner
warnings.filterwarnings("ignore")

# Reading in the labeled and unlabeled features
# Dropping all rows that have missing values. This is because we are using all columns as features for training our model
labeled_all_features = pd.read_csv('labeled_all_features.csv').dropna()
unlabeled_all_features = pd.read_csv('unlabeled_all_features.csv').dropna()

# Sorting out class labels. To convert the two column target into one. 
# MED = 1, UND = 1 is converted into class = 1
# MED = 1, UND = 0 is converted into class = 2
# MED = 0, UND = 1 is converted into class = 3
# MED = 0, UND = 0 is converted into class = 4

MEDs = list(labeled_all_features["MED"])
UNDs = list(labeled_all_features["UND"])

CLASS = []

class_dict = {1: 'MED: 1 and UND: 1', 2: 'MED: 1 and UND: 0', 3: 'MED: 0 and UND: 1', 4: 'MED: 0 and UND: 0'}

for i, val in enumerate(MEDs):
    if(MEDs[i] == 1 and UNDs[i] == 1):
        CLASS.append(1)
    elif(MEDs[i] == 1 and UNDs[i] == 0):
        CLASS.append(2)
    elif(MEDs[i] == 0 and UNDs[i] == 1):
        CLASS.append(3)
    elif(MEDs[i] == 0 and UNDs[i] == 0):
        CLASS.append(4)

# Dropping the old target columns: MED and UND. Their information has been encoded into the column "class"
labeled_all_features = labeled_all_features.drop(columns = ['MED', 'UND'])
labeled_all_features['class'] = CLASS

# Dropping some columns in the unlabeled dataset that were not present in the labeled dataset. 
# This is to ensure we are running our models on the same set of features.
lst_labeled_cols = list(labeled_all_features.columns)
old_unlabeled_all_features = unlabeled_all_features
unlabeled_all_features = unlabeled_all_features.drop(columns = ['cosine_similarity', 'accredited', 'mer_count_x'])
unlabeled_all_features = unlabeled_all_features.rename(columns={"duration": "video_duration", "mer_count_y": "medical_entity_terms"})
# unlabeled_all_features['class'] = -1

# Some pre-processing (Converting boolean to int for training)
labeled_all_features_np = labeled_all_features.to_numpy()
labeled_all_features_np[:, 2] = labeled_all_features_np[:, 2].astype(int)
labeled_all_features_np[:, 3] = labeled_all_features_np[:, 3].astype(int)
labeled_all_features_np[:, 4] = labeled_all_features_np[:, 4].astype(int)

unlabeled_all_features_np = unlabeled_all_features.to_numpy()
unlabeled_all_features_np[:, 2] = unlabeled_all_features_np[:, 2].astype(int)
unlabeled_all_features_np[:, 3] = unlabeled_all_features_np[:, 3].astype(int)
unlabeled_all_features_np[:, 4] = unlabeled_all_features_np[:, 4].astype(int)

# all_features_combined_np = np.vstack((labeled_all_features_np, unlabeled_all_features_np))

if __name__ == '__main__':
	X = labeled_all_features_np[:, 2:24]
	y = labeled_all_features_np[:, -1]
	y = y.astype('int')


	# Splitting into training and testing datasets
	# Not fixing a random_state because I ran this multiple times to get an average score based on different random_states
	# to ensure the results were more reliable.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Train-test split of 80-20

	
	print ('\nExtra Trees Classifier CoTraining')
	lg_co_clf = SelfTrainingClassifier(ExtraTreesClassifier())
	lg_co_clf.fit(X_train, y_train)
	y_pred = lg_co_clf.predict(X_test)
	print (classification_report(y_test, y_pred))

	print ('\nHistogram Gradient Boosting Trees Classifier CoTraining')
	lg_co_clf = SelfTrainingClassifier(HistGradientBoostingClassifier(max_iter = 100))
	lg_co_clf.fit(X_train, y_train)
	y_pred = lg_co_clf.predict(X_test)
	print (classification_report(y_test, y_pred))

	print ('\nAdaBoostClassifier CoTraining')
	lg_co_clf = SelfTrainingClassifier(AdaBoostClassifier())
	lg_co_clf.fit(X_train, y_train)
	y_pred = lg_co_clf.predict(X_test)
	print (classification_report(y_test, y_pred))

	print ('\nGradient Boosting Classifier CoTraining - n_estimators = 100')
	lg_co_clf = SelfTrainingClassifier(GradientBoostingClassifier(n_estimators = 100))
	lg_co_clf.fit(X_train, y_train)
	y_pred = lg_co_clf.predict(X_test)
	print (classification_report(y_test, y_pred))

	print ('\nRandom Forest Classifier CoTraining - max-depth = 15, n_estimators = 200')
	lg_co_clf = SelfTrainingClassifier(RandomForestClassifier(max_depth = 15, n_estimators = 200))
	lg_co_clf.fit(X_train, y_train)
	y_pred = lg_co_clf.predict(X_test)
	print (classification_report(y_test, y_pred))

