# Application of deep feedforward neural network for fraud detection in credir-card transactions using TensorFlow 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_curve
import math
import random

creditcard = pd.read_csv("../creditcard.csv")
creditcard = shuffle(creditcard)

# train dev test split
creditcard_train = creditcard.iloc[:200000,:]
creditcard_dev = creditcard.iloc[200000:244807,:]
creditcard_test = creditcard.iloc[-40000:,:]



# load the pandas dataframe as np array
train = creditcard_train.as_matrix()
dev = creditcard_dev.as_matrix()
test = creditcard_test.as_matrix()
(m,n) = train.shape

# oversample positive examples on the training set
train_P = train[train[:,-1] == 1]
train_P = train_P.repeat(int(200000/500),axis=0)

train = np.append(train,train_P,axis=0)
train=shuffle(train)

# Load and preprocess the data set
# train/dev/test split 200000/44807/40000
train_size = 200000

# extract the X and Y train and dev sets
train_X = train[:200000,:n-1].T
train_Y = train[:200000,-1:].T
dev_X = dev[:,:n-1].T
dev_Y = dev[:,-1:].T
test_X = test[:,:n-1].T
test_Y = test[:,-1:].T

# check dimensions are correct
print(train_X.shape)
print(train_Y.shape)
print(dev_X.shape)
print(dev_Y.shape)
print(test_X.shape)
print(test_Y.shape)

# number of positives in each set
print(np.sum(train_Y,axis=1))
print(np.sum(dev_Y,axis=1))
print(np.sum(test_Y,axis=1))

def prediction(X,parameters=None,threshold=0.5):
    # returns integer classifier predicted by the parameters

    x = tf.placeholder("float", shape=X.shape)
    keep_prob = tf.placeholder(tf.float32)
    Z = forward_propagation(x, parameters,keep_prob)
    prediction = tf.sigmoid(Z)

    with tf.Session() as SESS:
        pred = prediction.eval(feed_dict={x: X,keep_prob:1.0}, session=SESS)

    low_values_flags = pred < threshold  # Where values are low
    pred[low_values_flags] = 0
    high_values_flags = pred >= threshold  # Where values are low
    pred[high_values_flags] = 1
    return pred.flatten()

def prediction_in_sess(X,sess,parameters=None,threshold=0.9,kp=1.):
    # returns integer classifier predicted by the parameters
    x = tf.placeholder("float", shape=X.shape)
    keep_prob = tf.placeholder(tf.float32)
    Z = forward_propagation(x, parameters,kp)
    prediction = tf.sigmoid(Z)

    pred = prediction.eval(feed_dict={x: X,keep_prob:kp}, session=sess)

    low_values_flags = pred < threshold  # Where values are low
    pred[low_values_flags] = 0
    high_values_flags = pred >= threshold  # Where values are low
    pred[high_values_flags] = 1
    return pred.flatten()

def precision_recall(X,Y,parameters=None,threshold=0.9, verbal = 0,insess =0, sess=None):

    # function to compute precision and recall of a given model.
    # passing insess = 1 as well as a tf session to "sess" will run the prediction in the
    # existing session. For tracking precision during gradient descent and choosing
    # optimal early stopping time.

    # verbal = 0 silences output (for use in tf session)
    if verbal == 1:
        print("=======================")
        print("At threshold", threshold)

    if insess == 0:
        pred = prediction(X,parameters,threshold)
    else:
        pred = prediction_in_sess(X,sess, parameters,threshold)
    positives = 0
    true_positive = 0
    precision = 0
    recall = 0
    for i in range(0,len(Y)):
        if pred[i] > 0:
            positives += 1
            if Y[i] == 1:
                true_positive += 1
    if positives > 0:
        precision = true_positive/positives
        if verbal == 1:
            print("Precision:", precision)
    elif verbal == 1:
        print("No positive classifications!")

    # this will be an error if there are no positive examples in Y
    recall = true_positive/np.sum(Y)

    if verbal == 1:
        actual_positives = np.sum(Y)
        actual_negatives = len(Y) - actual_positives

        print("Recall at threshold:", recall)
        print("Number of true positives:", int(true_positive))
        print("Number of false positives:", int( positives-true_positive))
        print("Number of true negatives:", int(actual_negatives - (positives-true_positive)))
        print("Number of false negatives:", int(actual_positives - true_positive))
        print("F1 score:", 2*(recall*precision)/(recall+precision))

    return precision, recall


def area_under_prc( X, Y, parameters = None, verbal = 0, insess = 0, sess = None):

    # function computes area under the precision-recall curve using trapezoid
    # Riemann sums.
    # NOTE: the accuracy of this estimate varies a lot depending on the curve.

    partition_size = 150
    riemann_sum = 0
    integral_thresholds = np.arange(0,1,1/partition_size)
    recall = np.empty(partition_size+1)
    precision = np.empty(partition_size+1)

    # starting a tensorflow session is expensive, so we do it once and pass the session to
    # precision_recall, which speeds up the computation
    with tf.Session() as SESS:
        #compute area under PR curve
        precision[0],recall[0] = precision_recall(X,Y.flatten(),parameters,integral_thresholds[0],verbal = 0,insess=1,sess= SESS)
        for i in range(1,partition_size):
            precision[i],recall[i] = precision_recall(X,Y.flatten(),parameters,integral_thresholds[i],verbal = 0,insess=1,sess= SESS)
            riemann_sum += abs((recall[i]-recall[i-1])*(precision[i]+precision[i-1])/2)

    # add rectangle between last pr plot point and xy axis
    # this is a very rough estimate, as t approaches 1, P,R approach zero but the area lost is typically
    # very small.

    recall[partition_size] = 0
    precision[partition_size] = precision[partition_size-1]
    riemann_sum += abs((recall[partition_size]-recall[partition_size-1])*(precision[partition_size]+precision[partition_size-1])/2)

    # display results
    if verbal == 1:
        print("AUPRC:",riemann_sum)

        plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
          riemann_sum))
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()

    return

def lrelu(x, alpha=0.1):
    # leaky relu function
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def initialize_parameters():
    # layer sizes: 30 -> 20 -> 20 -> 20 -> 10 -> 1

    W1 = tf.get_variable("W1", [20,30], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [20,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [20,20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [20,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [20,20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [20,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [10,20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [10,1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [1,10], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b5 = tf.get_variable("b5", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5}

    return parameters

def forward_propagation(X, parameters, keep_prob):

    # forward prop: relu -> relu -> relu -> sigmoid

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = lrelu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = lrelu(Z2)
    # dropout layer
    Z3 = tf.add(tf.matmul(W3,tf.nn.dropout(A2, keep_prob)),b3)
    A3 = lrelu(Z3)
    Z4 = tf.add(tf.matmul(W4,A3),b4)
    A4 = lrelu(Z4)
    Z5 = tf.add(tf.matmul(W5,A4),b5)

    return Z5

def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.transpose(Y),logits = tf.transpose(Z)))
    return cost


def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32,shape=(n_x,None))
    Y = tf.placeholder(tf.float32,shape=(n_y,None))

    return X, Y

def random_mini_batches(X, Y, mini_batch_size, seed):

    np.random.seed(seed)
    m = X.shape[1]  # number of training examples
    n = Y.shape[0]  # dimension of one_hot vectors
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((n,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size*num_complete_minibatches : m]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*num_complete_minibatches : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(X_train, Y_train, X_dev, Y_dev, starter_learning_rate = 0.001,
          num_epochs = 200, minibatch_size = 10000, print_cost = True,beta=0.0001):

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    num_minibatches = int(m / minibatch_size)
    keep_prob = tf.placeholder(tf.float32)

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    x = tf.placeholder("float", shape=X_dev.shape)
    y = tf.placeholder("float", shape=Y_dev.shape)
    thresholds = np.arange(0,1,10,dtype='float32')

    # Initialize parameters
    parameters = initialize_parameters()

    # tensorflow graph
    Z = forward_propagation(X, parameters,keep_prob)
    cost = compute_cost(Z,Y)
    cost+= beta*tf.norm(parameters["W1"], ord = 'euclidean')
    cost+= beta*tf.norm(parameters["W2"], ord = 'euclidean')
    cost+= beta*tf.norm(parameters["W3"], ord = 'euclidean')
    cost+= beta*tf.norm(parameters["W4"], ord = 'euclidean')

    # gradient decay
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                       100000, 0.86, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost,global_step=global_step)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_epochs):

            epoch_cost = 0
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y,keep_prob : 0.5})
                epoch_cost += minibatch_cost / num_minibatches

            costs.append(epoch_cost)

        parameters = sess.run(parameters)

        # plot the cost
        plt.plot(np.squeeze(costs[5:]))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.show()

    return parameters



parameters_run_5 = model(train_X, train_Y, dev_X, dev_Y,num_epochs = 800)

area_under_prc(train_X,train_Y,parameters_run_5, verbal = 1)

area_under_prc(dev_X,dev_Y,parameters_run_5, verbal = 1)

# Run #6
#
# Run #5 did better on the dev set than earlier runs. Why stop there? Let's go for more epochs.

parameters_run_6 = model(train_X, train_Y, dev_X, dev_Y,num_epochs = 2000)

area_under_prc(train_X,train_Y,parameters_run_6, verbal = 1)

area_under_prc(dev_X,dev_Y,parameters_run_6, verbal = 1)

# Final results
#
# Out of all the models above, Run #6 gave us the best parameters (we had the highest AUPRC on the dev set with the parameters from this run). It's possible that we could get even better results if we simply run even longer.
#
# Let's estimate AUPRC on the test set using the parameters from run #6.

area_under_prc(test_X,test_Y,parameters_run_6, verbal = 1)

# To check the exact number of TP's and FP's at threshold = 0.98.

for i in range(0,5):
    _,_ = precision_recall(test_X,test_Y.flatten(),parameters_run_6,.9+i*.02,verbal=1)


