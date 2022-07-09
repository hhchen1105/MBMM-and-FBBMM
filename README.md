# MBMM-and-FBBMM
* Multivariate Beta Mixture Model (MBMM)
* Flexible Bivariate Beta Mixture Model (FBBMM)

## Setup
Tested under Python 3.10.4 in Ubuntu.
Install the required packages by
```
$ pip install -r requirements.txt
```

## Datasets
The following files are under folder data/
* wine_2d.csv: Data from sklearn.datasets.load_wine() is reduced to 2-dimensions by AutoEncoder. Shape: (178, 2)
* breast_cancer_2d.csv: Data from sklearn.datasets.load_breast_cancer() is reduced to 2-dimensions by AutoEncoder. Shape: (569, 2)
* mnist_2d.csv: Data from MNIST dataset is reduced to 2-dimensions by CNN and AutoEncoder. Shape: (70000, 2) (including train and test data)

All AutoEncoder codes are also in the folder.
## Execution
Compare k-means, MeanShift, DBSCAN, Agglomerative Clustering, GMM, MBMM and FBBMM on different datasets.

It is normal for FBBMM training time to be longer.

### Synthetic dataset
* Comparing different clustering algorithms on toy datasets. Reference from [scikit-learn](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html).
```
$ python3 synthetic.py
```

### Wine dataset
* Comparing different clustering except FBBMM on original features dataset.
* Comparing different clustering on 2-dimensions features dataset.
* Clustering performance evaluation metrics: Accuracy, ARI(Adjusted Rand Index), AMI(Adjusted Mutual Information).
```
$ python3 wine.py
```

### Breast cancer dataset
* Comparing different clustering except FBBMM on original features dataset.
* Comparing different clustering on 2-dimensions features dataset.
* Clustering performance evaluation metrics: Accuracy, ARI(Adjusted Rand Index), AMI(Adjusted Mutual Information).
```
$ python3 breast_cancer.py
```

### MNIST dataset
* Comparing different clustering on 2-dimensions feature dataset. Two Experinments: number0 and number3, number1 and number9
* Clustering performance evaluation metrics: Accuracy, ARI(Adjusted Rand Index), AMI(Adjusted Mutual Information).
```
$ python3 MNIST.py
```
