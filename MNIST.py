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

from keras.datasets import mnist

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
    data = pd.read_csv('data/mnist_2d.csv').to_numpy() #70000,2
    
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    lower, upper = 0.01, 0.99
    data = lower + (data - np.min(data))*(upper-lower)/(np.max(data)-np.min(data))
    target = np.concatenate((Y_train, Y_test), axis = 0)
    return data, target

def data_to_target(data, target):
    data_dict = {}
    for i in range(len(data)):
        if target[i] not in data_dict.keys():
            data_dict[target[i]] = []
            data_dict[target[i]].append(data[i])
        else:
            data_dict[target[i]].append(data[i])
            
    return data_dict


def smaple_number(num_list, data_dict):
    data = []
    target = []
    total_train = 0
    for n in num_list:  
        data = np.append(data, data_dict[n])        
        target = np.append(target, [n for _ in range(len(data_dict[n]))])       
        total_train += len(data_dict[n]) #累積的資料數 
    data = np.reshape(np.array(data), (total_train, 2))
    target = np.reshape(np.array(target), (total_train)).astype(int)
    
    return data, target
    

def initial_param(data, target):
    # ============
    # Initial parameters
    # ============
    MBMM_param19 = np.array([[1.09588370e+06, 1.42465917e+02, 4.05474548e+04],
                       [9.06447149e+04, 4.95003602e-02, 6.29934962e+05]])
    

    FBBMM_param19 = np.array([[3.66642644, 7.54841211, 0.10043114, 1.14688753],
                               [1.21178247, 1.26329   , 0.10032135, 4.33861118]])
    
    MBMM_param03 = np.array([[2.3892336,  9.91178983, 6.82128495],
                     [2.50212478, 8.67081669, 3.0153562 ]])
    
    FBBMM_param03 = np.array([[3.01042924, 1.15970268, 3.15086881, 0.10015249],
                               [0.10014359, 0.98369911, 3.29852921, 0.8037978 ]])
    
    data_dict = data_to_target(data, target)
    
    data_03, target_03 = smaple_number([0,3], data_dict)
    data_19, target_19 = smaple_number([1,9], data_dict)
     
    param = [(data_03, {'n_clusters': 2, 'quantile': .22, 'eps': .16, 'min_samples': 4920,'linkage': "ward", 
                     'affinity': "euclidean", 'MBMM_param': MBMM_param03, 'FBBMM_param':FBBMM_param03}),
            (data_19, {'n_clusters': 2, 'quantile': .51, 'eps': .011, 'min_samples': 5, 'linkage': "ward", 
                       'affinity': "euclidean", 'MBMM_param': MBMM_param19, 'FBBMM_param':FBBMM_param19})]
    target = [target_03, target_19]
    
    return param, target


if __name__ == "__main__":
    mnist_data, mnist_target = load_data()
               
    parameters, target = initial_param(mnist_data, mnist_target)

    for i_dataset, (dataset, params) in enumerate(parameters):  

        kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
               
        bandwidth = cluster.estimate_bandwidth(dataset, quantile=params['quantile'])
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

        dbscan = cluster.DBSCAN(eps=params['eps'], min_samples=params['min_samples'])

        aggolmarative = cluster.AgglomerativeClustering(
            linkage=params['linkage'],
            affinity=params['affinity'],
            n_clusters=params['n_clusters'],
        )

        gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

        mbmm = MBMM(n_components=params['n_clusters'], n_runs=100, param=params['MBMM_param'], tol=1e-3)
        
        fbbmm = FBBMM(n_components=params['n_clusters'], n_runs=20, param=params['FBBMM_param'], tol=1e-2)   

        clustering_algorithms = (
            ('K-means', kmeans),
            ("MeanShift", ms),
            ("DBSCAN", dbscan),
            ("AgglomerativeClustering", aggolmarative),
            ('GMM', gmm),
            ('MBMM', mbmm),
            ('FBBMM', fbbmm))
        
#         if i_dataset == 0:
#             target = target_03            
#         if i_dataset == 1:
#             target = target_19
            
        #print result
        if i_dataset == 0:     
            print('number0 and 3:')
        if i_dataset == 1:
             print('number1 and 9:')
                
        for name, algorithm in clustering_algorithms:
            algorithm.fit(dataset)

            if hasattr(algorithm, 'labels_'):
                train_predict_y = algorithm.labels_.astype(int)
            else:
                train_predict_y = algorithm.predict(dataset)        
           
            cluster_labels = infer_cluster_labels(train_predict_y, target[i_dataset])
            train_predicted_labels = infer_data_labels(train_predict_y, cluster_labels)       
            acc = np.round(np.count_nonzero(target[i_dataset] == train_predicted_labels)/len(target[i_dataset]), 3)

            ari_value = np.round(metrics.adjusted_rand_score(target[i_dataset], train_predict_y), 3)

            ami_value = np.round(metrics.adjusted_mutual_info_score(target[i_dataset], train_predict_y), 3)
                   
            print(name, {'Accuracy':acc, 'ARI':ari_value, "AMI":ami_value})
