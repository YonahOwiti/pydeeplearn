import argparse
import os
import fnmatch
import cv2
import sys
import random
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser(description=("make database"))
parser.add_argument('--net_file',
                     help="file where the serialized network should be saved",
                     type=str,
                     default="trained_net.p")
parser.add_argument('--path_to_data',
                     help="The path to where the training data is",
                     type=str,
                     default="")
parser.add_argument('--emotions',
                    nargs='+',
                    help='The emotions to be used to train the network.',
                    type=str)
parser.add_argument('--display_example_data',
                    dest='display_example_data',
                    action='store_true',
                    default=False,
                    help=("if true, shows a couple of the training examples."))
parser.add_argument('--cv',
                    dest='cv',
                    action='store_true',
                    default=False,
                    help=("if true, runs the cv code to try multiple hyperparameters "
                          "to find one which does best."))
args = parser.parse_args()

sys.path.append("..")

from lib import activationfunctions
from lib import deepbelief as db
import CNN as cnn
from lib import common
from keras.utils import np_utils



def getFiles(cateogry, show=False):
  """Returns all the png files in the subdirectory given by the input argument.

  The images are returns as a list of numpy arrays.
  """
  imgs = []

  extension = "png"
  path = os.path.join(args.path_to_data, cateogry)

  print "reading files from", path

  imageFiles = [(os.path.join(dirpath, f), f)
        for dirpath, dirnames, files in os.walk(path)
        for f in fnmatch.filter(files, '*.' + extension)]

  for fullPath, shortPath in imageFiles:
    img = cv2.imread(fullPath, 0)
    print img.reshape(-1).shape
    imgs += [img.reshape(-1)]
    if show:
      plt.imshow(img, cmap=plt.cm.gray)
      plt.show()

  return imgs


def createTrainingSet(show=False):
  # Read the unsupervised data.
  unsupervised = getFiles("unsupervised")
  unsupervised = np.array(unsupervised)

  img_rows, img_cols = 48, 48
  img_channels = 1
  global nb_classes
  nb_classes = 7

  XTrain=pickle.load(open('XNormTrain2','rb'))
  yTrain=pickle.load(open('yTrain','rb'))
  XTest=pickle.load(open('XNormValid2','rb'))
  yTest=pickle.load(open('yValid','rb'))

  X_train=np.reshape(XTrain,(len(XTrain),1,48,48))
  y_train=np.array(yTrain)
  X_test=np.reshape(XTest,(len(XTest),1,48,48))
  y_test=np.array(yTest)


  # the data, shuffled and split between train and test sets
  #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
  print('X_train shape:', X_train.shape)
  print('X_test shape:', X_test.shape)
  print(X_train.shape[0], 'train samples')
  print(X_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  Y_train = np_utils.to_categorical(y_train, nb_classes)
  Y_test = np_utils.to_categorical(y_test, nb_classes)

  small_index = list(np.random.choice(XTrain.shape[0], 10000, replace=False))
  XTrain_small = XTrain[small_index,:]
  Y_train_small = Y_train[small_index,:]

  data = XTrain_small
  labels = Y_train_small

  return unsupervised, data, labels


def visualizeTrainingData():
  unsupervised, data, labels = createTrainingSet()

  if unsupervised:
    print 'show unsupervised data'
    for i in xrange(5):
      plt.imshow(unsupervised[i].reshape((40,30)), cmap=plt.cm.gray)
      plt.show()

  print 'showing supervised data'
  for i in xrange(20):
    plt.imshow(data[i].reshape((40,30)), cmap=plt.cm.gray)
    plt.show()


def trainNetWithAllData():
  unsupervisedData, data, labels = createTrainingSet()
  unsupervisedData = None
  img_rows, img_cols = 48, 48
  img_channels = 1
  nb_classes = 7

  activationFunction = activationfunctions.Rectified()
  rbmActivationFunctionVisible = activationfunctions.Identity()
  rbmActivationFunctionHidden = activationfunctions.RectifiedNoisy()

  unsupervisedLearningRate = 0.0001
  supervisedLearningRate = 0.001
  momentumMax = 0.99

  print "This is input data shape",data.shape
  print labels.shape

  net = db.DBN(4, [2304, 1500, 1000, nb_classes],
             binary=False,
             activationFunction=activationFunction,
             rbmActivationFunctionVisible=rbmActivationFunctionVisible,
             rbmActivationFunctionHidden=rbmActivationFunctionHidden,
             unsupervisedLearningRate=unsupervisedLearningRate,
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=True,
             rbmNesterovMomentum=True,
             rmsprop=True,
             miniBatchSize=20,
             hiddenDropout=0.5,
             visibleDropout=0.8,
             momentumFactorForLearningRateRBM=False,
             firstRBMheuristic=False,
             rbmVisibleDropout=1.0,
             rbmHiddenDropout=1.0,
             preTrainEpochs=10,
             sparsityConstraintRbm=False,
             sparsityRegularizationRbm=0.001,
             sparsityTragetRbm=0.01)

  net.train(data, labels, maxEpochs=200,
            validation=False,
            unsupervisedData=unsupervisedData)


  with open(args.net_file, "wb") as f:
    pickle.dump(net, f)
  return net


def trainAndTestNet():
  unsupervisedData, data, labels = createTrainingSet()

  print np.unique(np.argmax(labels, axis=1))

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  print data
  data = common.scale(data)
  unsupervisedData = None

  activationFunction = activationfunctions.Rectified()
  rbmActivationFunctionVisible = activationfunctions.Identity()
  rbmActivationFunctionHidden = activationfunctions.RectifiedNoisy()

  unsupervisedLearningRate = 0.0001
  supervisedLearningRate = 0.001
  momentumMax = 0.99

  trainData = data[train]
  trainLabels = labels[train]

  net = db.DBN(4, [2304, 1500, 1000, nb_classes],
             binary=False,
             activationFunction=activationFunction,
             rbmActivationFunctionVisible=rbmActivationFunctionVisible,
             rbmActivationFunctionHidden=rbmActivationFunctionHidden,
             unsupervisedLearningRate=unsupervisedLearningRate,
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=True,
             rbmNesterovMomentum=True,
             rmsprop=True,
             miniBatchSize=20,
             hiddenDropout=0.5,
             visibleDropout=0.8,
             momentumFactorForLearningRateRBM=False,
             firstRBMheuristic=False,
             rbmVisibleDropout=1.0,
             rbmHiddenDropout=1.0,
             preTrainEpochs=10,
             sparsityConstraintRbm=False,
             sparsityRegularizationRbm=0.001,
             sparsityTragetRbm=0.01)

  net.train(trainData, trainLabels, maxEpochs=10,
            validation=False,
            unsupervisedData=unsupervisedData)

  probs, predicted = net.classify(data[test])


  actualLabels = labels[test]
  correct = 0
  errorCases = []

  for i in xrange(len(test)):
    actual = actualLabels[i]
    print probs[i]
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0 / len(test)

  confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)
  print "confusion matrix"
  print confMatrix

  with open(args.net_file, "wb") as f:
    pickle.dump(net, f)
  return net

# Performs CV to choose the best hyperparameters given the data.
def getHyperParamsAndBestNet():
  unsupervisedData, data, labels = createTrainingSet()

  print np.unique(np.argmax(labels, axis=1))

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  print data
  data = common.scale(data)
  unsupervisedData = None

  activationFunction = activationfunctions.Rectified()
  rbmActivationFunctionVisible = activationfunctions.Identity()
  rbmActivationFunctionHidden = activationfunctions.RectifiedNoisy()

  tried_params = []
  percentages = []
  best_index = 0
  index = 0
  best_correct = 0

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=10)
  for train, test in kf:
    unsupervisedLearningRate = random.uniform(0.0001, 0.2)
    supervisedLearningRate = random.uniform(0.0001, 0.2)
    momentumMax = random.uniform(0.7, 1)

    tried_params += [{'unsupervisedLearningRate': unsupervisedLearningRate,
                      'supervisedLearningRate': supervisedLearningRate,
                      'momentumMax': momentumMax}]

    trainData = data[train]
    trainLabels = labels[train]

    net = db.DBN(4, [1200, 1500, 1000, len(args.emotions)],
               binary=False,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=True,
               rbmNesterovMomentum=True,
               rmsprop=True,
               miniBatchSize=20,
               hiddenDropout=0.5,
               visibleDropout=0.8,
               momentumFactorForLearningRateRBM=False,
               firstRBMheuristic=False,
               rbmVisibleDropout=1.0,
               rbmHiddenDropout=1.0,
               preTrainEpochs=10,
               sparsityConstraintRbm=False,
               sparsityRegularizationRbm=0.001,
               sparsityTragetRbm=0.01)

    net.train(trainData, trainLabels, maxEpochs=200,
              validation=False,
              unsupervisedData=unsupervisedData)

    probs, predicted = net.classify(data[test])

    actualLabels = labels[test]
    correct = 0

    for i in xrange(len(test)):
      actual = actualLabels[i]
      print probs[i]
      if predicted[i] == np.argmax(actual):
        correct += 1

    percentage_correct = correct * 1.0 / len(test)
    print "percentage correct"
    print percentage_correct

    if percentage_correct > best_correct:
      best_index = index
      best_correct = percentage_correct
      with open(args.net_file, "wb") as f:
        pickle.dump(net, f)

    percentages += [percentage_correct]
    index += 1

  print 'best params'
  print tried_params[best_index]
  print 'precision'
  print best_correct


if __name__ == '__main__':
    #trainNetWithAllData()
    trainAndTestNet()
