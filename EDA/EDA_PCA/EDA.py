import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def plot_2d(data_2d, ylabels, labels, y_offset=-0.001):
    # print(ylabels)
    scatter = plt.scatter(data_2d[:,0], data_2d[:, 1], c=ylabels, cmap = 'turbo')
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.title('Principal Component Analysis')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def plot_2d_new(data_2d, ylabels, labels, y_offset=-0.001):
    # print(ylabels)
    scatter = plt.scatter(data_2d[:,0], data_2d[:, 1], c=ylabels, cmap = 'turbo')
    # plt.legend(handles=scatter.legend_elements()[0])
    plt.title('K-Means clustering results')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

labeled_EDA = pd.read_csv('labeled_EDA.csv')
# cols = list(labeled_EDA.columns)[2:-1]
labeled_EDA = labeled_EDA.iloc[: , 2:]
cols = list(labeled_EDA.columns)[2:-1]
corr1 = labeled_EDA['transcript_sentence_count'].corr(labeled_EDA['transcript_unique_wordcount'])
print('Correlation between transcript_sentence_count and transcript_unique_wordcount', corr1)

# print(cols)
# print('len(cols):',len(cols))
# print(labeled_EDA)
data = labeled_EDA.to_numpy()
data_X = data[:, 2:]*1
# print(data_X)

labels = data_X[:, -1]
# print(labels)
X = data_X[:, :-1]
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
# print(scaled_data)
X = scaled_data
# print(len(X[0]))

# tsne = TSNE(n_components=2, perplexity=55, learning_rate=0.01, n_iter=5000, init='pca', verbose=1, random_state=0)
# transformed_X = tsne.fit_transform(X)
# print(len(transformed_X))
# print(transformed_X)
# plot_2d(transformed_X, labels)

# One-dimensional PCA plotted against category

single_dimension_pca = PCA(n_components=1)
pca_X_single = single_dimension_pca.fit_transform(X)
labels_y = ['MED = 0, UND = 0', 'MED = 0, UND = 1', 'MED = 1, UND = 0', 'MED = 1, UND = 1']
scatter = plt.scatter(pca_X_single, labels, c=labels, cmap = 'turbo')
plt.legend(handles=scatter.legend_elements()[0], labels=labels_y)
plt.title('Principal Component Analysis')
plt.xlabel('Component 1')
plt.ylabel('Category')
plt.show()

# Two-dimensional PCA

double_dimension_pca = PCA(n_components=2)  # project data down to two dimensions
labels_y = ['MED = 0, UND = 0', 'MED = 0, UND = 1', 'MED = 1, UND = 0', 'MED = 1, UND = 1']
# print(len(pca_X)) # 521

pca_X = double_dimension_pca.fit_transform(X)
plot_2d(pca_X, labels, labels_y) # uncomment for different graph

kmeans = KMeans(n_clusters=4, random_state=0).fit(pca_X)
kmeans_preds = kmeans.predict(pca_X)
kmeans_labels = kmeans.labels_
plot_2d_new(pca_X, kmeans_labels, labels_y)


fig = plt.figure()
plt.bar(cols, double_dimension_pca.components_[0])
fig.subplots_adjust(bottom=0.55)
plt.xticks(rotation=90)
plt.title('Weights of Component 1 with corresponding columns')
plt.ylabel('Weight')
plt.show()

fig = plt.figure()
plt.bar(cols, double_dimension_pca.components_[1])
fig.subplots_adjust(bottom=0.55)
plt.xticks(rotation=90)
plt.title('Weights of Component 2 with corresponding columns')
plt.ylabel('Weight')
plt.show()

# plot_2d(pca_X, labels, labels_y) # uncomment for different graph

labels_reshaped = labels.reshape(len(pca_X), 1)
combined_pca_labels = np.hstack((pca_X, labels_reshaped))

combined_pca_labels_1 = combined_pca_labels[combined_pca_labels[:, -1] == 1]
combined_pca_labels_2 = combined_pca_labels[combined_pca_labels[:, -1] == 2]
combined_pca_labels_3 = combined_pca_labels[combined_pca_labels[:, -1] == 3]
combined_pca_labels_4 = combined_pca_labels[combined_pca_labels[:, -1] == 4]

combined_pca_labels_1_coords = combined_pca_labels_1[:, :-1]
combined_pca_labels_2_coords = combined_pca_labels_2[:, :-1]
combined_pca_labels_3_coords = combined_pca_labels_3[:, :-1]
combined_pca_labels_4_coords = combined_pca_labels_4[:, :-1]

combined_pca_labels_1_coords_avg = np.mean(combined_pca_labels_1_coords, axis = 0)
combined_pca_labels_2_coords_avg = np.mean(combined_pca_labels_2_coords, axis = 0)
combined_pca_labels_3_coords_avg = np.mean(combined_pca_labels_3_coords, axis = 0)
combined_pca_labels_4_coords_avg = np.mean(combined_pca_labels_4_coords, axis = 0)

coords = [combined_pca_labels_1_coords_avg,
          combined_pca_labels_2_coords_avg, 
          combined_pca_labels_3_coords_avg, 
          combined_pca_labels_4_coords_avg]

coords = np.array(coords)
labels_coords = [1, 2, 3, 4]
labels = ['MED = 1, UND = 1', 'MED = 1, UND = 0', 'MED = 0, UND = 1', 'MED = 0, UND = 0']


scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels_coords)
plt.title('Average Coordinates for each class after Principal Component Analysis')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.annotate('('+str(round(coords[0][0],2))+','+str(round(coords[0][1],2))+')', (27,65), fontsize = 7)
plt.annotate('('+str(round(coords[1][0],2))+','+str(round(coords[1][1],2))+')', (24,-262), fontsize = 7)
plt.annotate('('+str(round(coords[2][0],2))+','+str(round(coords[2][1],2))+')', (-87,-82), fontsize = 7)
plt.annotate('('+str(round(coords[3][0],2))+','+str(round(coords[3][1],2))+')', (-50,118), fontsize = 7)

plt.legend(handles=scatter.legend_elements()[0], labels=labels)
plt.show()