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

    def get_sample(self, sample_count):
        distribution_mean = np.random.random_sample(2) * (self.maxx, self.maxy)  # considering mins are 0

        eigvals = np.random.random_sample(2)
        eigvals *= (self.maxx / self.distribution_variance, self.maxy/self.distribution_variance)
        eigvals **= 2

        diagonal_matrix = np.diag(eigvals)  # this ?

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

    def sample_gauss_2d(self, C, N):
        X = []
        Y = []

        for distr in range(C):
            X.extend(self.get_sample(N))
            Y.extend([distr]*N)

        return np.array(X), np.array(Y)


def eval_perf_binary(Y, Y_):
    true_positives = sum(np.logical_and(Y == Y_, Y_ is True))
    false_negatives = sum(np.logical_and(Y != Y_, Y_ is True))
    true_negatives = sum(np.logical_and(Y == Y_, Y_ is False))
    false_positives = sum(np.logical_and(Y != Y_, Y_ is False))

    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    accuracy = (true_positives + true_negatives) / (true_positives+false_negatives + true_negatives+false_positives)

    return accuracy, recall, precision


def eval_AP(ranked_labels):
    positives = np.sum(ranked_labels)
    negatives = len(ranked_labels) - positives

    true_positives = positives
    true_negatives = 0
    false_negatives = 0
    false_positives = negatives

    precision_sum = 0

    for label in ranked_labels:
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)

        if label:
            precision_sum += precision

        true_positives -= label
        false_negatives += label
        false_positives -= not label
        true_negatives += not label

    return float(precision_sum) / float(positives)


if __name__ == "__main__":
    G = Random2DGaussian()
    X = G.get_sample(100)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
