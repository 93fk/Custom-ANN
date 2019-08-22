# coding: utf-8

"""
Converting MNIST dataset to .csv format
"""

NAME = '0_MNIST_to_csv' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'Custom_ANN'
PYTHON_VERSION = '3.6.8'

# Preamble

## Imports
"""
All the Python imports go here.
"""

import os, re
import pandas as pd
from mnist import MNIST

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
dataset = MNIST(os.path.join('empirical', '0_data', 'external'))

train_images, train_labels = dataset.load_training()
test_images, test_labels = dataset.load_testing()

train = pd.DataFrame(train_images)
train['labels'] = train_labels

test = pd.DataFrame(test_images)
test['labels'] = test_labels

train.to_csv(os.path.join(pipeline, 'out', 'train.csv'))
test.to_csv(os.path.join(pipeline, 'out', 'test.csv'))