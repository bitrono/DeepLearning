import loglinear as ll
import random
from collections import Counter
import numpy as np

STUDENT = {'name': 'Ofir Bitron',
           'ID': '200042414'}
I2L = {}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    mat = np.zeros(input_size)
    for bigram in features:
        mat[F2I.get(bigram)] += 1
    return mat


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        f = feats_to_vec(features)
        y = L2I.get(label)
        if ll.predict(f, params) == y:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            vec = feats_to_vec(features)  # convert features to a vector.
            y = L2I.get(label)  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(vec, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            params = [params[0] - grads[0] * learning_rate, params[1] - grads[1] * learning_rate]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def text_to_unigrams(text):
    return list(text)
	
def I2L():
	'''
	Create dictionary with id of language as key, and language as value.
	'''	
	for key, value in L2I.items():
		I2l[value] = key
	
def read_test_file():
	I2L()
	with open(r"C:\Users\bitro\OneDrive\שולחן העבודה\university\deep learning\test", "r") as rf:
		with open(r"C:\Users\bitro\OneDrive\שולחן העבודה\university\deep learning\out_test", "w") as wf:
			text = rf.readline()
			while text != '':
				data = text.split('\t')[1]
				prediction_num = ll.predict(feats_to_vec(text_to_bigrams(data)))
				wf.write(I2L[prediction_num] + '\n')
				

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    global input_size
    ################UNIGRAMS################# train- 0.7363 dev-0.6666
    # train_data = [(l, text_to_unigrams(t)) for l, t in read_data("train")]
    # dev_data = [(l, text_to_unigrams(t)) for l, t in read_data("dev")]
    # learning_rate = 0.0016
    # input size= 30
    #################BIGRAMS#################  train- 0.907. dev-0.86333.

    train_data = [(l, text_to_bigrams(t)) for l, t in read_data("train")]
    dev_data = [(l, text_to_bigrams(t)) for l, t in read_data("dev")]
    learning_rate = 0.00085
    input_size = 600
    #########################################
    fc = Counter()
    for l, feats in train_data:
        fc.update(feats)
    # 600 most common bigrams in the training set.
    global vocab
    vocab = set([x for x, c in fc.most_common(input_size)])
    # label strings to IDs
    global L2I
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_data]))))}

    # feature strings (bigrams) to IDs
    global F2I
    F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}

    num_iterations = 30
    in_dim = input_size
    out_dim = 6
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
