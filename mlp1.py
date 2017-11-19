import numpy as np
import loglinear as ll
import pdb

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
	W, b, U, b_tag = params
	calc_tanh = np.vectorize(tanh)
	return ll.softmax(np.dot(calc_tanh(np.dot(x, W) + b), U) + b_tag)

def tanh(x):
	return ((np.e ** (2 * x)) - 1) / ((np.e ** (2 * x)) + 1)
	
def predict(x, params):
    return np.argmax(classifier_output(x, params))
    
def create_y_vector(y, dim_of_vector):
    temp_vec = np.zeros(dim_of_vector)
    temp_vec[y] = 1
    return temp_vec

def loss_and_gradients(x, y, params):

    W, b, U, b_tag = params
    calc_tanh = np.vectorize(tanh)
    probs = classifier_output(x, params)
    y_hat = np.dot(calc_tanh(np.dot(x, W) + b), U) + b_tag
    y_vec = create_y_vector(y, len(y_hat))
    
    # Create gradient matrices.
    gb = np.zeros(b.shape)
    gW = np.zeros(W.shape)
    gU = np.zeros(U.shape)
    gb_tag = np.zeros(b_tag.shape)

    # Compute gradient according to U.
    gU = np.outer(tanh(np.dot(x, W) + b), (y_hat - y_vec)) 
        
    # Compute gradient of b_tag
    gb_tag = y_hat - y_vec

    # Compute dot product.
    dot_prod = np.dot(U, (y_hat - y_vec))
    
    # Compute the gradient according to b.        
    gb =  dot_prod * (1 - (tanh(np.dot(x, W) + b) ** 2))
    
    # Compute the gradient according to W.
    gW = np.outer(x, gb)
          
    # Compute loss.	
    loss = -1 * np.log(probs[y])
    return loss, [gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    xavier = 6 ** 0.5
    
    # Compute the initial value range.
	glW = (xavier) / (hid_dim + in_dim)
	glB = (xavier) / (hid_dim + 1)

	glU = (xavier) / (hid_dim + out_dim)
	glb_tag = (xavier) / (out_dim + 1)

	W = np.random.uniform(-glW, glW, (in_dim, hid_dim))
	b = np.random.uniform(-glB, glB, (hid_dim))
	U = np.random.uniform(-glU, glU, (hid_dim, out_dim))
	b_tag = np.random.uniform(-glb_tag, glb_tag, (out_dim))

	return [W, b, U, b_tag]

