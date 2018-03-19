#! /usr/bin/env python

import numpy as np

from data import Random2DGaussian

"""
def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_loss(prob, y):
    """
    Sauce: https://en.wikipedia.org/wiki/Cross_entropy
    """
    return -y * np.log(prob) - (1 - y) * np.log(1 - prob)


def binlogreg_train(X, Y_, param_niter=100000, param_delta=0.1):
    '''
    Arguments
      X:  data, np.array NxD
      Y_: class indexes, np.array Nx1

    Returns
      w, b: logistic regression parameters
    '''
    samples_count = len(X)
    
    w = np.random.randn(2)
    b = 0

    for i in range(param_niter):
        # scores with current weights and biases
        scores = np.dot(X,w) + b

        # probs for our results
        probs  = sigmoid(scores)

        # loss = np.sum(-1 * np.log(probs)) # is zis the same? or should I combine with Y_
        # idiot
        loss = np.sum(cross_entropy_loss(probs, Y_))

        if i % 10 == 0:
            print("Iteration {}: loss {}".format(i, loss))

        dl_ds = scores - Y_  # Y_ are labels, I believe they can be understood as booleans

        grad_w = 1./samples_count * np.dot(dl_ds * X) 
        grad_b = 1./samples_count * np.sum(dl_ds) 

        w += -1 * param_delta * grad_w
        b += -1 * param_delta * grad_b


def binlog_classify(X, w, b):
'''
  Argumenti
      X:    podatci, np.array NxD
      w, b: parametri logističke regresije

  Povratne vrijednosti
      probs: vjerojatnosti razreda c1
'''
    return sigmoid(np.dot(X,w) + b)


if "__name__" == "__main__":
    G = Random2DGaussian()
    X = G.get_sample(100)


