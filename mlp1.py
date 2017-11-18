import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
	W, b, U, b_tag = params
    calc_tanh = np.vectorize(tanh)
	return ll.softmax(np.dot(U, calc_tanh(np.dot(W, x) + b)) + b_tag)

def tanh(x):
	return ((np.e ** (2 * x)) - 1) / ((np.e ** (2 * x)) + 1)
	
def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):

	W, b, U, b_tag = params
	calc_tanh = np.vectorize(tanh)
    probs = classifier_output(x, params)
	y_hat = np.dot(U, calc_tanh(np.dot(W, x) + b)) + b_tag
	
	# Create gradient matrices.
	gb = np.zeros(b.shape)
	gW = np.zeros(W.shape)
	gU = np.zeros(U.shape)
	gb_tag = np.zeros(b_tag.shape)
	
	it_U = np.nditer(gb, flags=["multi_index"], op_flags=["readwrite"])
	# Compute the gradient according to U.
	while not it_U.finished:
		j = it_U.multi_index[1]
		i = it_U.multi_index[0]
		gU[i][j] = (y_hat[j] - y[j]) / tanh(np.dot(W, x) + b)[i]
		it_U.next()
		
	itb_tag = np.nditer(gb, flags=["multi_index"], op_flags=["readwrite"])	
	# Compute the gradient according to b_tag.
	while not itb_tag.finished:
		j = itb_tag.multi_index[1]
		i = itb_tag.multi_index[0]
		gb_tag[i] = y_hat[i] - y[i]
		itb_tag.next()
	
	
	it_b = np.nditer(gb, flags=["multi_index"], op_flags=["readwrite"])
	# Compute the gradient according to b.
	while not it_b.finished:
		j = it_b.multi_index[1]
		i = it_b.multi_index[0]
		gb[j] = (y_hat - y) * U[i] * (1 - tanh(np.dot(W, x) + b)[i]) ** 2)
		it_b.next()
		
	it_W = np.nditer(gb, flags=["multi_index"], op_flags=["readwrite"])
	# Compute the gradient according to W.
	while not it_W.finished:
		j = it_b.multi_index[1]
		i = it_b.multi_index[0]
		gW[i][j] = ((y_hat - y) * U[j] * (1 - tanh(np.dot(W, x) + b)[i]) ** 2) / x[i])
		it_W.next()
		
		
    return ...

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
	
	glW = (6 ** 0.5) / (hid_dim + in_dim)
	glB = (6 ** 0.5) / (hid_dim + 1)

	glU = (6 ** 0.5) / (hid_dim + out_dim)
	glb_tag = (6 ** 0.5) / (out_dim + 1)

	W = np.random.uniform(-glW, glW, (hid_dim, in_dim))
	b = np.random.uniform(-glB, glB, (hid_dim))
	U = np.random.uniform(-glU, glU, (out_dim, hid_dim))
	b_tag = np.random.uniform(-glb_tag, glb_tag, (out_dim))
	
    return [W, b, U, b_tag]
	
