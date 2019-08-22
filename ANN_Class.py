
# coding: utf-8

"""
Class for building custom ANN
"""

NAME = '0_ANN_Class'
PROJECT = 'Custom_ANN'
PYTHON_VERSION = '3.6.8'

## Imports
import os, re
import numpy as np
import random

# ---------
# Main code
# ---------
class ANN():

    def __init__(self, *layers):
        """
        Create an Artificial Neural Network given nodes number in each layer.
        First number is the number of nodes in the input layer and the last is the number of nodes in the output layer.

        Given the set of numbers: 100, 50, 30, 20, 5 we expect vector of length 100 as an input, and vector of length 5 as an output.
        Numbers: 50, 30 and 20 are the number of nodes in hidden layers.

        The network uses ReLU as an activation function in intermediate layer and Softmax function in the output layer, therefore
        the network is aimed to work well with crassification problems.
        """
        self.layers = len(layers) - 1
        w = 0.05 # initialized weights and biases range from -w to w
        self.biases = [np.random.uniform(low=-w, high=w, size=layer) for layer in layers[1:]]
        self.weights = [np.random.uniform(low=-w, high=w, size=(next_layer, prev_layer))\
            for prev_layer, next_layer in zip(layers[:-1], layers[1:])]

    def forward_run(self, input_vector):
        """
        Given the input vector, the method will return the output as well as intermediate layers output, on given ANN object.
        """
        z_vectors = []
        for layer in range(self.layers):
            if layer == self.layers - 1:
                z = self.weights[layer].dot(hidden_layer) + self.biases[layer]
                z_vectors.append(z)
                output_vector = softmax(z) # last layer
            elif layer == 0:
                z = self.weights[layer].dot(input_vector) + self.biases[layer]
                z_vectors.append(z)
                hidden_layer = ReLU(z) # first layer
            else:
                z = self.weights[layer].dot(hidden_layer) + self.biases[layer]
                z_vectors.append(z)
                hidden_layer = ReLU(z) # middle layers
        return output_vector, z_vectors

    def back_propagation(self, input_vector, target):
        """
        This method will propagate errors and adjust weights and biases to minimize prediciton error.
        """
        eta = 0.00001 # learning rate
        output_vector, z_vectors = self.forward_run(input_vector)
        sm_d = softmax_derivative(z_vectors[-1])
        error = (output_vector - target)*sm_d # delta_L
        self.biases[-1] = self.biases[-1] - error*eta # bias_L
        self.weights[-1] = self.weights[-1] - np.outer(error, ReLU(z_vectors[-2]))*eta # weights_L
        for i in range(self.layers - 1, 1, -1):
            error = (self.weights[i].T.dot(error))*ReLU_derivative(z_vectors[i-1])
            self.biases[i-1] = self.biases[i-1] - error*eta
            self.weights[i-1] = self.weights[i-1] - np.outer(error, ReLU(z_vectors[i-2]))*eta
        error = (self.weights[1].T.dot(error))*ReLU_derivative(z_vectors[0])
        self.biases[0] = self.biases[0] - error*eta
        self.weights[0] = self.weights[0] - np.outer(error, input_vector)*eta
        pass

# Activation functions

def softmax(z):
    e = np.exp(z - np.max(z))
    return e/np.sum(e)

def ReLU(z):
    return np.maximum(z, 0)

def softmax_derivative(z):
    s = softmax(z)
    return s*(1-s)

def ReLU_derivative(z):
    return np.maximum(np.sign(z), 0)

