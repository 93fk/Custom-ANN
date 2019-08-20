from mnist import MNIST
import pandas as pd

dataset = MNIST('/home/filip/Git/Custom ANN/MNIST/')

train_images, train_labels = dataset.load_training()
test_images, test_labels = dataset.load_testing()

def min_max(x):
    return (x - min(x))/(max(x) - min(x))

train = pd.DataFrame(train_images)
train['labels'] = train_labels

test = pd.DataFrame(test_images)
test['labels'] = test_labels

test.to_csv('/home/filip/Git/Custom ANN/MNIST/test.csv')
train.to_csv('/home/filip/Git/Custom ANN/MNIST/train.csv')