import numpy as np
import os 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.metrics import homogeneity_score

labelspath = '..\Autopack\example_usage\outputs\default_motifs_opt_True.csv'
labelslist = pd.read_csv(labelspath)
labeldict = dict(zip(labelslist.mol, labelslist.pred))

imagepath = "..\Autopack\PAH\PAH .npy"
imagelist = os.listdir(imagepath)
list1=[]
list2=[]
imagelist = [i for i in imagelist if i.endswith('npy')]

for i in imagelist:
    fullimagepath = os.path.join(imagepath, i)
    imagedata = np.load(fullimagepath)
    imagedata1 = np.nan_to_num(imagedata)
    
    #calculates dft for each pixel outputting a 90x90 matrix with fourier equation in each cell
    DFT = np.fft.fft2(imagedata1)
    DFT_real = np.real(DFT)
    flatten_matrix = DFT_real.flatten()
    list1.append(flatten_matrix)
    #DFT_imag = np.imag(DFT)
    
    refcode = i.strip('.npy')
    motif = labeldict[refcode]
    list2.append(motif)

motif_label = np.array(list2)
FFT_data = np.array(list1)

#is there information loss when doing this step?
#FFT_data1 = np.nan_to_num(FFT_data)

#splitting data into training and test sets        
training_data, testing_data = train_test_split(FFT_data, test_size=0.2, random_state=25) 
training_label_data, testing_label_data = train_test_split(motif_label, test_size=0.2, random_state=25) 
print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")
print(f"No. of training label examples: {training_label_data.shape[0]}")
print(f"No. of testing label examples: {testing_label_data.shape[0]}")


#standardize data
scaler = StandardScaler()
scaler.fit(training_data)
scaler.fit(testing_data)
training_data = scaler.transform(training_data)
testing_data = scaler.transform(testing_data)    

#PCA
pca = PCA(.70)
#pca = PCA(n_components=4)
pca.fit(training_data)
variance = pca.explained_variance_ratio_

var = np.cumsum(np.round(variance, 3)*100)
plt.figure(figsize=(12,6))
plt.ylabel('% Variance Explained', fontsize=20)
plt.xlabel('# of Features', fontsize=20)
plt.title('PCA Analysis', fontsize=20)
plt.ylim(0,100.5)
plt.plot(var)

pca.n_components_
print("Number of components before PCA  = " + str(FFT_data.shape[1]))
print("Number of components after PCA  = " + str(pca.n_components_)) #dimension reduced from 90
training_data = pca.transform(training_data)
testing_data = pca.transform(testing_data)
print("Dimension of our train data after PCA  = " + str(training_data.shape))
print("Dimension of our test data after PCA  = " + str(testing_data.shape))

#pca_df_scale = pd.DataFrame(training_data, columns=['pc1','pc2','pc3','pc4'])
print("PCA explained varience ratio", pca.explained_variance_ratio_)

# # Plot the explained variances
# features = range(pca.n_components_)
# plt.bar(features, pca.explained_variance_ratio_, color='black')
# plt.xlabel('PCA features')
# plt.ylabel('variance %')
# plt.xticks(features)

# Save components to a DataFrame
PCA_components = pd.DataFrame(training_data)

# plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')

# # elbow method to check clusters
# ks = range(1, 10)
# inertias = []
# for k in ks:
#     # Create a KMeans instance with k clusters: model
#     model = KMeans(n_clusters=k)
    
#     # Fit model to samples
#     model.fit(PCA_components.iloc[:,:3])
    
#     # Append the inertia to the list of inertias
#     inertias.append(model.inertia_)
    
# plt.plot(ks, inertias, '-o', color='black')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(ks)
# plt.show()

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 35)
k_means.fit(training_data)
k_means_labels = k_means.labels_ #List of labels of each dataset

labels = k_means.predict(training_data)

print("The list of labels of the clusters are " + str(np.unique(k_means_labels)))
clusters_pca_scale = pd.concat([PCA_components, pd.DataFrame({'pca_clusters':k_means_labels})], axis=1)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
 
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(training_label_data, labels))
print("Completeness: %0.3f" % metrics.completeness_score(training_label_data, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(training_label_data, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(training_label_data, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(training_label_data, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(training_data, labels))

plt.figure(figsize = (10,10))
sns.scatterplot(clusters_pca_scale.iloc[:,0],clusters_pca_scale.iloc[:,1], hue=k_means_labels, palette='Set1', s=100, alpha=0.2).set_title('KMeans Clusters (4) Derived from PCA', fontsize=15)
plt.legend()
plt.show()

# #Different plot
# db = DBSCAN(eps=0.3, min_samples=10).fit(training_data)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True

# # Plot result
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)

#     xy = training_data[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)

#     xy = training_data[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()







    
    
