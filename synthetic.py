#Code reference: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


import MBMM
from MBMM import MBMM
import FBBMM
from FBBMM import FBBMM

np.random.seed(0)


def load_data_and_param():
    from sklearn import cluster, datasets, mixture
    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.3,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    #upper left to lower right
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    #upper right to lower left
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[-0.6, -0.6], [0.4, 0.8]]
    X_aniso2 = np.dot(X, transformation)
    aniso2 = (X_aniso2, y)


    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)
    

    MBMM_param = [np.array([[0.5,0.5,2.],[6.,6.,6.]]),
                 np.array([[0.5,0.5,2.],[6.,6.,6.]]),
                 np.array([[6.,8.,6.],[2.,10.,9.],[20.,20.,8.]]),
                 np.array([[1.,5.,2.],[6.,6.,6.],[3.,3.,6.]]),    
                 np.array([[2.,11.,3.],[11.,2.,3.],[4.,0.8,1.3]]),
                 np.array([[2.,9.,3.],[9.,2.,3.],[8.,8.,8.]])]

    FBBMM_param = [np.array([[0.8,0.8,0.8,0.8],[2.,2.,2.,2.]]),
                 np.array([[2.,8.,2.,2.],[2.,2.,8.,2.]]),
                 np.array([[2.,2.,2.,8.],[4.,2.,2.,2.],[4.,4.,4.,4.]]),
                 np.array([[2.,2.,2.,5],[2.,2.8,2.,2.],[5.,2.,2.,2.]]),    
                 np.array([[4.,2.,4.,2.],[8.,8.,2.,2.],[2.,2.,2.,4.]]),
                 np.array([[4.,2.,4.,1.],[2.,4.,2.,2.],[2.,2.,2.,4.]])]
    
    default_base = {'quantile': .2,
                    'eps': .05,
                    'damping': .9,
                    'preference': -5,
                    'n_neighbors': 8,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1,
                    'threshold': 0.05,
                    'MBMM_param': MBMM_param,
                    'FBBMM_param': FBBMM_param}

    datasets = [(noisy_circles, {'damping': .85, 'preference': -20,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),
                (noisy_moons, {'damping': .86, 'preference': -6, 'n_clusters': 2,'quantile': .25}),
                (varied, {'eps': .05, 'n_neighbors': 2,
                          'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2, 'quantile': .3}),
                (aniso, {'eps': .03, 'n_neighbors': 2,
                         'min_samples': 20, 'xi': 0.04, 'min_cluster_size': .2}),
                (aniso2, {'eps': .03, 'n_neighbors': 2,
                         'min_samples': 20, 'xi': 0.04, 'min_cluster_size': .2}),
                (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": .2})]
    
    return datasets, default_base


if __name__ == "__main__":
    
    
    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=((9 * 2 + 3)*0.7, 13/6*6))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    plot_num = 1

    datasets, default_base = load_data_and_param()

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        lower, upper = 0.01, 0.99
        X = lower + (X - np.min(X))*(upper-lower)/(np.max(X)-np.min(X))

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

        two_means = cluster.KMeans(n_clusters=params['n_clusters'])

        dbscan = cluster.DBSCAN(eps=params["eps"])

        aggolmarative = cluster.AgglomerativeClustering(
            linkage="average",
            affinity="cityblock",
            n_clusters=params["n_clusters"],
            connectivity=connectivity,
        )

        gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

        mbmm = MBMM(n_components = params['n_clusters'], n_runs = 100, param=params['MBMM_param'][i_dataset], tol = 1e-3)

        fbbmm = FBBMM(n_components = params['n_clusters'], n_runs = 20, param=params['FBBMM_param'][i_dataset], tol = 1e-2)

        clustering_algorithms = (
            ('K-means', two_means),
            ("MeanShift", ms),
            ("DBSCAN", dbscan),
            ("Agglomerative\nClustering", aggolmarative),
            ('GMM', gmm),
            ('MBMM', mbmm),
            ('FBBMM', fbbmm)
        )

        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1
       
    plt.show()

