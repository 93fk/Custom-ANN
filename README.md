# Project description

Artificial Neural Networks gain popularity each day. It seems it is essential for any Data Scientist nowadays to know how they operate. On the other hand, it is taken for granted to use standard ANN libraries like Keras or PyTorch to build one, but it doesn't quite show you the math behind it.

It occurred to me there is no better way to learn what's going on under the hood than to build one from scratch. Therefore I searched for some books that explain the math and I found 'Neural Networks and Deep Learning' by Michael Nielsen. It's an online book, that from my point of view, can teach you everything to build a functioning ANN from the ground up.

Let's take a look at the results!

# Building the ANN

What surprised me the most is how little elements you need in order to build an ANN. In my case just this few building blocks were sufficient to call my project a success:

* ANN Initialization,
* Forward Run Mechanism,
* Backpropagation Mechanism,
* Activation Functions.

## ANN initialization

Typical ANN is defined by the number of layers and the number of nodes in each layer. I opted for the approach where the only input (`*layers` argument) is the number of nodes in each layer, including input and output layers. In this way, all we need is a couple of integers to initialize the ANN!

```python
def __init__(self, *layers):
```

Given the information we've got, we can easily initialize weights and biases - the heart of the ANN.

```python
w = 0.05 # initialized weights and biases range from -w to w
self.biases = [np.random.uniform(low=-w, high=w, size=layer) for layer in layers[1:]]
self.weights = [np.random.uniform(low=-w, high=w, size=(next_layer, prev_layer))\
            for prev_layer, next_layer in zip(layers[:-1], layers[1:])]
```

As you can see the **biases** are vectors of the same length as the hidden layers (including the output layer), whereas the **weights** are matrices of dimensions $m \times n$ where *m* is the length of the vector in the *current* layer and the *n* is the length of the vector in the *previous* layer.

## Activation Functions

As I am aiming to deal with multiclass classification problem in this project I decided to use Softmax function in the output layer, and the ReLU function in the hidden layers, as it is simple to implement and quite effective (no vanishing gradient problem as in sigmoid or tanh functions).

```python
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
```

I also implemented their derivatives, as they are essential in the backpropagation algorithm.

## Forward Run

The core principle of the ANN operation is the forward run mechanism. It takes the current layer, takes a dot product with the weights, adds biases and applies activation function to get activations in the next layers.

```python
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
```

The above method acts on an instantiated neural network and takes an input vector to get predictions for classification. The *z_vector* is used only in the backpropagation.

## Backpropagation

The most complex part of the project was the implementation of the learning algorithm: backpropagation. This is essentially a derivative of the cost function with respect to the weights and biases.

First, we need to calculate the prediction error in the output layer, which is the derivative of the cost function with respect to the output times the derivative of the activation function (softmax).

```python
sm_d = softmax_derivative(z_vectors[-1])
error = (output_vector - target)*sm_d
```

The cost fuction was $C(x)=\frac{1}{2}(t-x)^{2}$, where *t* is the target value and *x*  is the predicted value, therefore the derivative is simply a subtraction.

Error, in any layer that is not an output layer, depends on the next layer, as well as on the weights and the derivative of the activation function (in our case it's ReLU). The interpretation of the equation is that the error is propagated backward (from the output to the input layer) - that's why it's called backpropagation.

```python
error = (self.weights[i].T.dot(error))*ReLU_derivative(z_vectors[i-1])
```

Then all that's left is to update weights and biases, where *eta* is the learning rate.

```python
self.biases[i-1] = self.biases[i-1] - error*eta
self.weights[i-1] = self.weights[i-1] - np.outer(error, ReLU(z_vectors[i-2]))*eta
```

Full code for the backpropagation:

```python
def back_propagation(self, input_vector, target):
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
```

# The Results

The ANN was trained on MNIST dataset. It achieves 90.7% accuracy on the test set.

# If you are looking for AI development services, check out: [https://neoteric.eu/services/ai-development/](https://neoteric.eu/services/ai-development/)

![](https://github.com/93fk/Custom-ANN/blob/master/empirical/2_pipeline/2_Visualize_ANN/out/ConfusionMatrix.png?raw=true)

On the below animation you can see how the net operates. On the left side, you can see the input image, and nex to it you have 1st hidden layer, 2nd hidden layer, and the output layer. You can clearly see how the ReLU activation function 'shuts down' certain neurons and how the softmax function votes for the most probable class.

![](https://github.com/93fk/Custom-ANN/blob/master/empirical/2_pipeline/2_Visualize_ANN/out/ANN_visualized.gif?raw=true)
