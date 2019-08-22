import ANN_Class as ANN
import numpy as np
import random

nn = ANN.ANN(30, 20 ,10, 5)

input_ = np.array([random.random() for i in range(30)])
target = np.array([1 if i == 2 else 0 for i in range(5)])
t, _ = nn.forward_run(input_)
t
for i in range(100000):
    nn.back_propagation(input_, target)
    t, _ = nn.forward_run(input_)
    t
t