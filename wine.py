from sklearn import cluster, datasets, mixture
import numpy as np
import pandas as pd

import MBMM
from MBMM import MBMM
import FBBMM
from FBBMM import FBBMM

import random
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import warnings

## Calculate accuracy reference from: https://github.com/sharmaroshan/MNIST-Using-K-means  
def infer_cluster_labels(pred_labels, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """
    
    inferred_labels = {}
    n_clusters = len(set(actual_labels))
    #n_clusters = len(np.unique(pred_labels))
    for i in range(n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(pred_labels == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if len(counts) > 0:        
            if np.argmax(counts) in inferred_labels:
                # append the new number to the existing array at this slot
                inferred_labels[np.argmax(counts)].append(i)
            else:
                # create a new array in this slot
                inferred_labels[np.argmax(counts)] = [i]    

    return inferred_labels  
    
    
def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    predicted_labels = np.array([-1 for i in range(len(X_labels))])
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

def load_data():
    my_dataset = datasets.load_wine()
    data = my_dataset.data
    target = my_dataset.target

    data_2d = pd.read_csv('data/wine_2d.csv').to_numpy() 

    lower, upper = 0.01, 0.99
    data = lower + (data - np.min(data))*(upper-lower)/(np.max(data)-np.min(data))
    data_2d = lower + (data_2d - np.min(data_2d))*(upper-lower)/(np.max(data_2d)-np.min(data_2d))
    return data, data_2d, target

def initial_param(data, data_2d):
    # ============
    # Initial parameters
    # ============
    MBMM_param = np.array([[16.04, 10.65, 10.51, 19.67, 65.07, 10.34, 10.07, 9.45, 9.99, 12.21, 9.7 , 10.46, 627.21, 873.58],
                             [0.99, 0.74, 0.76, 1.06, 2.86, 0.77, 0.77, 0.71, 0.75, 0.82, 0.73, 0.77, 73.88, 30.81],
                             [13.57, 9.07, 9.01, 17.25, 51.63, 8.95, 8.84, 8.17, 8.69, 9.73, 8.43, 9.15, 272.17, 749.45]])
    MBMM_2d_param = np.array([[30526.07727405, 195165.66397019, 157390.63851422],                    
                           [222081.43959903, 199765.57227599,  91559.15760877],
                           [258114.89230211,  31131.40037692,  78670.34794596]])

    FBBMM_2d_param = np.array([[6.38624405, 3.01039041, 2.73489288, 0.1],
                               [1.60944964, 0.1       , 4.40972969, 5.12624954],
                               [3.40052985, 9.46763675, 0.1       , 3.49745015]])
    
    param = [(data, {'n_clusters': 3, 'quantile': 0.303, 'eps': .015, 'linkage': "complete", 
                     'affinity': "manhattan", 'MBMM_param': MBMM_param}),
            (data_2d, {'n_clusters': 3, 'quantile': 0.303, 'eps': .086, 'linkage': "average", 
                       'affinity': "l2", 'MBMM_param': MBMM_2d_param, 'FBBMM_param':FBBMM_2d_param})]

    return param


if __name__ == "__main__":
    data, data_2d, target = load_data()
    
    parameters = initial_param(data, data_2d)
    
    for i_dataset, (dataset, params) in enumerate(parameters):  

        kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
               
        bandwidth = cluster.estimate_bandwidth(dataset, quantile=params['quantile'])
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

        dbscan = cluster.DBSCAN(eps=params['eps'])

        aggolmarative = cluster.AgglomerativeClustering(
            linkage=params['linkage'],
            affinity=params['affinity'],
            n_clusters=params['n_clusters'],
        )

        gmm = mixture.GaussianMixture(n_components=params['n_clusters'])

        mbmm = MBMM(n_components=params['n_clusters'], n_runs=100, param=params['MBMM_param'], tol=1e-3)
        
        clustering_algorithms = [
            ('K-means', kmeans),
            ("MeanShift", ms),
            ("DBSCAN", dbscan),
            ("AgglomerativeClustering", aggolmarative),
            ('GMM', gmm),
            ('MBMM', mbmm)       
        ]
               
        if i_dataset == 1:
            fbbmm = FBBMM(n_components=params['n_clusters'], n_runs=20, param=params['FBBMM_param'], tol=1e-2)   
            clustering_algorithms.append(('FBBMM', fbbmm))
        
        #print result
        if i_dataset == 0:     
            print('Original data:')
        else:
             print('2-dim data:')
                
        for name, algorithm in clustering_algorithms:

            algorithm.fit(dataset)

            if hasattr(algorithm, 'labels_'):
                train_predict_y = algorithm.labels_.astype(int)
            else:
                train_predict_y = algorithm.predict(dataset)        

            cluster_labels = infer_cluster_labels(train_predict_y, target)
            train_predicted_labels = infer_data_labels(train_predict_y, cluster_labels)       
            acc = np.round(np.count_nonzero(target == train_predicted_labels)/len(target), 3)

            ari_value = np.round(metrics.adjusted_rand_score(target, train_predict_y), 3)

            ami_value = np.round(metrics.adjusted_mutual_info_score(target, train_predict_y), 3)
                   
            print(name, {'Accuracy':acc, 'ARI':ari_value, "AMI":ami_value})
