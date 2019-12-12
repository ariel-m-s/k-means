import numpy as np
import numpy.random as rd


def normalize(value, min_, max_):
    return (value - min_) / (max_ - min_)


def resize(value, min_, max_):
    return value * (max_ - min_) + min_


def get_resized_centroid(ranges):
    values = rd.rand(len(ranges))
    return list(map(resize, values, ranges))


def get_centroids(n, size):
    return [list(rd.rand(size)) for _ in range(n)]


def get_closest_centroid(vector, centroids):
    return min(centroids, key=lambda x: np.linalg.norm(vector - x))
