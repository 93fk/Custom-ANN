import numpy as np

class ANN():

    def __init__(self, *layers):
        self.layers = len(layers) - 1
        self.biases = [np.random.uniform(low=-5, high=5, size=layer) for layer in layers[1:]]
        self.weights = [np.random.uniform(low=-5, high=5, size=(next_layer, prev_layer))\
            for prev_layer, next_layer in zip(layers[:-1], layers[1:])]

    def forward_run(self, input_vector):
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
        output_vector, z_vectors = self.forward_run(input_vector)
        error = (output_vector - target)*softmax_derivative(z_vectors[-1]) # delta_L
        print(error)
        tmp.biases[-1] = tmp.biases[-1] - error # bias_L
        tmp.weights[-1] = tmp.weights[-1] - np.outer(error, ReLU(z_vectors[-2])) # weights_L
        for i in range(self.layers - 1, 1, -1):
            error = (self.weights[i].T.dot(error))*ReLU_derivative(z_vectors[i-1])
            tmp.biases[i-1] = tmp.biases[i-1] - error
            tmp.weights[i-1] = tmp.weights[i-1] - np.outer(error, ReLU(z_vectors[i-2]))
        error = (self.weights[1].T.dot(error))*ReLU_derivative(z_vectors[0])
        tmp.biases[0] = tmp.biases[0] - error
        tmp.weights[0] = tmp.weights[0] - np.outer(error, input_vector)
        pass

def softmax(z):
    e = np.exp(z - np.max(z))
    return e/np.sum(e)

def ReLU(z):
    return np.maximum(z, 0)

def softmax_derivative(z):
    s = z.reshape(-1,1)
    return (np.diagflat(s) - np.dot(s, s.T)).diagonal()

def ReLU_derivative(z):
    return np.maximum(np.sign(z), 0)

target = np.array([0,1,0,0,0,0])
tmp = ANN(10,8,5,6)
input_ = np.array([i for i in range(10)])
t, _ = tmp.forward_run(input_)
t

tmp.back_propagation(input_, target)
t, _ = tmp.forward_run(input_)
t