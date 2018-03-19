#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:
    def __init__(self):
        self.minx = 0
        self.maxx = 10
        self.miny = 0
        self.maxy = 10

        self.distribution_variance = 5

        np.random.seed(100)

    def get_sample(self, sample_count):
        distribution_mean = np.random.random_sample(2) * (self.maxx, self.maxy) # considering mins are 0

        eigvals = np.random.random_sample(2)
        eigvals *= (self.maxx / self.distribution_variance, self.maxy/self.distribution_variance)
        eigvals **= 2

        diagonal_matrix = np.diag(eigvals) # this ?

        rot_matrix_angle = np.random.random_sample() * np.pi * 2

        rot_matrix = np.array(
            [
                [np.cos(rot_matrix_angle), - np.sin(rot_matrix_angle)],
                [np.sin(rot_matrix_angle), np.cos(rot_matrix_angle)]
            ]
        )

        sigma_matrix = np.dot(np.transpose(rot_matrix), diagonal_matrix, rot_matrix)

        sample = np.random.multivariate_normal(distribution_mean, sigma_matrix, sample_count)

        return sample


if __name__ == "__main__":
    G = Random2DGaussian()
    X = G.get_sample(100)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
