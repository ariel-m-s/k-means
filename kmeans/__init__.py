import numpy as np
from vega_datasets import data
from utils import get_centroids, get_closest_centroid, normalize


class KMeansModel:
    def __init__(self):
        self.clusters = None

    def update_clusters(self, source, n_clusters):
        self.clusters = {i: [] for i in range(n_clusters)}

        for vector in source:
            centroid = get_closest_centroid(vector, self.centroids)
            idx = self.centroids.index(centroid)

            self.clusters[idx].append(vector)

    def update_centroids(self):
        if not self.clusters:
            raise AttributeError("cannot update centroids.")

        self.centroids = list(list(np.mean(self.clusters[cluster], axis=0))
                              for cluster in self.clusters)

    def cluster(self, data, n_clusters, n_iter):
        source = data.to_numpy()
        self.centroids = get_centroids(n_clusters, source.shape[1])

        for _ in range(n_iter):
            self.update_clusters(source, n_clusters)
            self.update_centroids()

        return self.centroids, self.clusters


if __name__ == "__main__":
    source = data.seattle_weather().drop(columns=["date", "weather"])
    source = normalize(source, source.min(), source.max())
    model = KMeansModel()
    model.cluster(source, 4, 2)
