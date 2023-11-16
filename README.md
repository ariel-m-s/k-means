# K-Means clustering implementation in Python

## Overview

This repository contains a Python implementation of the K-Means clustering algorithm. K-Means is a widely used unsupervised machine learning algorithm for partitioning a dataset into a specified number (k) of clusters, based on similarity. This implementation utilizes Python 3 and Numpy for the numerical operations.

## Files

1. `kmeans.py`

This file contains the main implementation of the `KMeansModel` class, which encapsulates the K-Means clustering logic.

2. `utils.py`

This file provides utility functions used in the K-Means implementation. These functions include normalization, resizing, and centroid generation.

3. `main.ipynb`

Jupyter Notebook demonstrating the usage of the K-Means algorithm. It serves as a visual guide and provides insights into the clustering process.

## Example usage

```python
from kmeans import KMeansModel
import pandas as pd

# Load your data into a pandas DataFrame (replace this with your own data)
data = pd.read_csv("your_data.csv")

# Instantiate the KMeansModel
model = KMeansModel()

# Cluster the data
centroids, clusters = model.cluster(data, n_clusters=4, n_iter=10)
```

## Model parameters

- `n_clusters`: The number of clusters the algorithm partitions the dataset into.
- `n_iter`: The number of iterations the algorithm goes through to iteratively update cluster assignments and centroids.
