
# coding: utf-8

"""
Visualizing the ANN
"""

NAME = '2_Visualize_ANN'
PROJECT = 'Custom_ANN'
PYTHON_VERSION = '3.6.8'

## Imports

import os, re, time, pickle, imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from ANN_Class import ReLU, softmax
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

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
X_test = pd.read_csv(os.path.join('empirical', '2_pipeline',
                                  '0_MNIST_to_csv', 'out', 'test.csv'), header=0, index_col=0)
y_test = X_test['labels']
X_test = X_test.iloc[:,:-1]



Net = pickle.load(open(os.path.join('empirical', '2_pipeline', '1_Compile_ANN', 'out', 'Net.p'), 'rb'))

flat_images = X_test.iloc[[3, 2, 1, 18, 4, 15, 11, 0, 61, 7, 3]].values.flatten()
partial_images = []
predictions = []
for i in range(280):
    p_image = flat_images[i*28:i*28+784]
    partial_images.append(p_image.reshape(28,28))
    out, z = Net.forward_run(p_image)
    predictions.append((out, z))

results = []
for i in X_test.index:
    out, z = Net.forward_run(X_test.iloc[i])
    results.append(np.argmax(out))
results = np.array(results)

frames = []
plt.rcParams['image.cmap'] = 'inferno'

def plotting(iterator):
    fig = plt.figure(constrained_layout=True, figsize=(7,3))
    gs = GridSpec(1, 21, figure=fig)

    ax0 = fig.add_subplot(gs[:8])
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[8:16])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[16:20])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[20:21])
    ax3.set_xticks([])
    ax3.set_yticks([i for i in range(10)])

    img = partial_images[iterator]
    pred = predictions[iterator]

    ax0.imshow(img)
    ax1.imshow(ReLU(pred[1][0].reshape(16,16)))
    ax2.imshow(ReLU(pred[1][1].reshape(8,4)))
    ax3.imshow(pred[0].reshape(10,1))

    text = f'Prediction: {np.argmax(pred[0])}'
    fig.suptitle(text, fontsize=20)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

frames = []
for frame in range(280):
    if frame % 28 != 0:
        frames.append(plotting(frame))
    else:
        for i in range(10):
            frames.append(plotting(frame))


kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave(os.path.join(pipeline, 'out', 'ANN_visualized.gif'), frames, fps=20)



cm = confusion_matrix(y_test, results)
cm_df = pd.DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])

plt.figure(figsize=(6,6.5))
plt.suptitle('Confusion Matrix for MNIST data set predictions', fontsize=12)
ax = sn.heatmap(cm_df, annot=True, square=True, fmt="d", cbar=False)
ax.set(xlabel='Predicted', ylabel='Actual')

plt.savefig(os.path.join(pipeline, 'out', 'ConfusionMatrix.png'))

# ----------
# Leftovers
# ----------
"""
Here you leave any code snippets or temporary code that you don't need but don't want to delete just yet
"""