import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

# Loading in the data and dropping rows with missing values
labeled_all_features = pd.read_csv('labeled_all_features.csv', index_col = 0).dropna()
unlabeled_all_features = pd.read_csv('unlabeled_video_data.csv')
unlabeled_all_features = unlabeled_all_features.dropna()

labeled_all_features = labeled_all_features.drop(columns = ['OCR_confidence_avg', 'medical_entity_terms'])

# Sorting out class labels
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

labeled_all_features = labeled_all_features.drop(columns = ['MED', 'UND'])
labeled_all_features['class'] = CLASS

lst_labeled_cols = list(labeled_all_features.columns)

old_unlabeled_all_features = unlabeled_all_features
unlabeled_all_features = unlabeled_all_features.drop(columns = ['accredited', 'narrative_readability', 'description_MER_terms'])

cols_to_use = list(unlabeled_all_features.columns)

# Some pre-processing (Converting boolean to int for training)
labeled_all_features_np = labeled_all_features.to_numpy()
labeled_all_features_labels = labeled_all_features_np[:, -1]
labeled_all_features_np = labeled_all_features_np[:, :-1]
labeled_all_features_np[:, 2] = labeled_all_features_np[:, 2].astype(int)
labeled_all_features_np[:, 3] = labeled_all_features_np[:, 3].astype(int)
labeled_all_features_np[:, 4] = labeled_all_features_np[:, 4].astype(int)

unlabeled_all_features_np = unlabeled_all_features.to_numpy()
unlabeled_all_features_np[:, 2] = unlabeled_all_features_np[:, 2].astype(int)
unlabeled_all_features_np[:, 3] = unlabeled_all_features_np[:, 3].astype(int)
unlabeled_all_features_np[:, 4] = unlabeled_all_features_np[:, 4].astype(int)

# Model training - Ensemble
multiclass_classifiers = [
    KNeighborsClassifier(n_neighbors = 7),
    GaussianNB(),
    HistGradientBoostingClassifier(),
    AdaBoostClassifier(),
    RandomForestClassifier(),
    MLPClassifier(hidden_layer_sizes=(50, 50, 50),
                             max_iter=5000, activation='relu'),
    GradientBoostingClassifier()
]
multiclass_classifiers_names = [
    'KNeighborsClassifier() - K = 7',
    'GaussianNB()',
    'HistGradientBoostingClassifier()',
    'AdaBoostClassifier()',
    'RandomForestClassifier()',
    'MLP() - 50,50,50. num_epochs = 5000, activation = ReLU',
    'GradientBoostingClassifier()'
]

X = labeled_all_features_np[:, 1:]*1
y = labeled_all_features_labels.astype('int')
X_unlabeled = unlabeled_all_features_np[:, 1:]

skf = StratifiedKFold(n_splits=5)
for idx, clf in enumerate(multiclass_classifiers):
    print('Training classifier: ', multiclass_classifiers_names[idx])
    i = 0
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.05, random_state=35, stratify=y)
    f1_scores_split = []
    clfs = []
    for train_index, test_index in skf.split(X_train_all, y_train_all):
        print('Split: ', i+1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1_scores_split.append(f1_score(y_test, y_pred, average = 'micro'))
        clfs.append(clf)
        i += 1
    f1_scores_split = np.array(f1_scores_split)
    max_score_idx = np.argmax(f1_scores_split)
    max_clf = clfs[max_score_idx]
    y_pred = max_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('='*50)

# Stratifying the train-val split using a validation set size of 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)

# Training K-Means
kmeans = KMeans(n_clusters=4, random_state=0).fit(X_train)
kmeans_preds = kmeans.predict(X_test)
kmeans_labels = kmeans.labels_

all_predictions = []

sm = SMOTE()
X_res, y_res = sm.fit_resample(X_train, y_train)

clf = RandomForestClassifier()
clf.fit(X_res, y_res)
y_preds_probs = clf.predict_proba(X_unlabeled)
y_preds = clf.predict(X_unlabeled)

n = 0.7
probs_higher_than_70 = {}
probs_higher_than_60 = {}
probs_higher_than_50 = {}
for idx, row in enumerate(y_preds_probs):
    pred_row = y_preds_probs[idx]
    if(np.max(pred_row) > 0.7):
        probs_higher_than_70[idx] = (y_preds[idx], np.max(pred_row))
    if(np.max(pred_row) > 0.6):
        probs_higher_than_60[idx] = (y_preds[idx], np.max(pred_row))
    if(np.max(pred_row) > 0.5):
        probs_higher_than_50[idx] = (y_preds[idx], np.max(pred_row))

def print_link(idx, conf, unlabeled_all_features):
    print('With conf: ', conf, 'link: ', 'https://www.youtube.com/watch?v='+ unlabeled_all_features['video_ID'].iloc[idx])

for idx, (label, prob) in probs_higher_than_50.items():
    print_link(idx, '50%', unlabeled_all_features)
    print('Label = ', label)
    print('Prob', prob)
print('='*50)
for idx, (label, prob) in probs_higher_than_60.items():
    print_link(idx, '60%', unlabeled_all_features)
    print('Label = ', label)
    print('Prob', prob)
print('='*50)
for idx, (label, prob) in probs_higher_than_70.items():
    print_link(idx, '70%', unlabeled_all_features)
    print('Label = ', label)
    print('Prob', prob)

# from sklearn.metrics import f1_score
num_neighbors = range(1, 100)

# Finding best number of K for KNN
mean_validation_scores = []
test_scores = []

for n in num_neighbors:
    clf = KNeighborsClassifier(n_neighbors = n)
    scores = cross_validate(clf, X_res, y_res, cv=10, scoring='f1_macro', return_estimator = True)
    mean_score = np.mean(scores['test_score'])
    mean_validation_scores.append(mean_score)
    max_score = np.argmax(scores['test_score'])
    models = scores['estimator']
    best_model = models[max_score]
    y_pred = best_model.predict(X_test)
    test_score = f1_score(y_test, y_pred, average='micro')
    test_scores.append(test_score)

plt.plot(num_neighbors, mean_validation_scores, color = 'b', label = 'Mean Validation Score')
plt.plot(num_neighbors, test_scores, color = 'r', label = 'Test Score')
plt.title('Comparing validation and test scores for different number of neighbors in KNN')
plt.xlabel('Number of Neighbors')
plt.ylabel('F1-score')
plt.legend(loc='best')
plt.show()

all_predictions = []

for clf, name in zip(multiclass_classifiers, multiclass_classifiers_names):
    print('Showing results for:', name)
    y_pred = clf.predict(X_test)
    all_predictions.append(y_pred)

all_predictions = np.array(all_predictions)

# Ensemble of ensembles
ensemble_preds = []
for col_idx in range(len(all_predictions[0])):
    preds = all_predictions[:, col_idx]
    counts = np.bincount(preds).argmax()
    ensemble_preds.append(counts)

print('Ensemble of Ensembles Result: ')
print(classification_report(y_test, ensemble_preds))
