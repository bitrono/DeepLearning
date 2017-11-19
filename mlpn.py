import numpy as np

STUDENT = {'name': 'Ofir Bitron',
           'ID': '200042414'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    return ...

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    j = 0
    xavier = 6 ** 0.5
    params = []
    
    for i in len(dims) - 1:
        # Compute the initial value range.
        temp_matrix = xavier / (dims[i] + dims[i + 1])
        temp_vec = xavier / (dims[i + 1] + 1)
        
        # Create matrix 
        params[j] = np.random.uniform(-temp_matrix, temp_matrix, (dims[i], dims[i + 1]))
        
        # Create vector
        params[j + 1] = np.random.uniform(-temp_vec, temp_vec, dims[i + 1])
        
        j += 2
     
    return params
