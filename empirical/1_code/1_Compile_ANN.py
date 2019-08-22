
# coding: utf-8

"""
Compiling the ANN using MNIST dataset
"""

NAME = '1_Compile_ANN' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Custom_ANN'
PYTHON_VERSION = '3.6.8'

# Preamble

import os, re, random, pickle
import numpy as np
import pandas as pd
from ANN_Class import ANN
from progress.bar import Bar

## Set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)


## Set  up pipeline folder if missing
if os.path.exists(os.path.join('empirical', '2_pipeline')):
    pipeline = os.path.join('empirical', '2_pipeline', NAME)
else:
    pipeline = os.path.join('2_pipeline', NAME)

if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))

# ---------
# Main code
# ---------
random.seed('ANN')
Net = ANN(784,256,32,10)

X_train = pd.read_csv(os.path.join('empirical', '2_pipeline',
                                   '0_MNIST_to_csv', 'out', 'train.csv'), header=0, index_col=0)
y_train = X_train['labels']
X_train = X_train.iloc[:,:-1]

X_test = pd.read_csv(os.path.join('empirical', '2_pipeline',
                                  '0_MNIST_to_csv', 'out', 'test.csv'), header=0, index_col=0)
y_test = X_test['labels']
X_test = X_test.iloc[:,:-1]

def accuracy(net, test_set, test_labels):
    results = []
    for i in test_set.index:
        output, _ = net.forward_run(test_set.iloc[i])
        results.append(np.argmax(output))
    return sum(np.array(results) == test_labels.values)/test_labels.shape[0]


for epoch in range(8):
    bar = Bar('Processing', max=X_train.shape[0])
    for i in X_train.index:
        target = np.array([1 if j == y_train.iloc[i] else 0 for j in range(10)])
        Net.back_propagation(X_train.iloc[i], target)
        bar.next()
    acc = accuracy(Net, X_test, y_test)
    print(f'\tEpoch {epoch+1} accuracy: {round(acc*100, 2)}%')

pickle.dump(Net, open(os.path.join(pipeline, 'out', 'Net.p'), 'wb'))