import glob

import numpy as np
import pandas as pd

files = glob.glob('data/tests_*.csv')
d = pd.concat([pd.read_csv(f, sep=';', decimal=',', encoding='cp1251', header=None, skiprows=4, nrows=460) for f in files])

dd = d.iloc[:, 2:]
d_norm = (dd-dd.min())/(dd.max()-dd.min())

X = d_norm.iloc[:, 2:].to_numpy(dtype=np.float64)
L = d.iloc[:, 0 ].to_numpy()

L_unique = np.unique(L)

model_shape = [X.shape[1], 16, L_unique.shape[0]]
n_layers    = len(model_shape)

def init_weights_and_biases(shape):
	assert(len(shape)>1)

	weights = np.array([np.random.randn(shape[i+1], shape[i]) for i in range(len(shape)-1)], dtype=object)
	biases  = np.array([np.zeros(shape[i]) for i in range(1, len(shape))], dtype=object)
	return weights, biases

weights, biases = init_weights_and_biases(model_shape)

def produce_target_values(shape, labels, labels_set):
	assert(shape[0] == labels.shape[0])

	target = np.zeros(shape)
	for i in range(shape[0]):
		idx = np.where(labels_set == labels[i])
		target[i][idx] = 1.0

	return target

def mse(target, predicted):
	assert(target.shape == predicted.shape)

	return np.sum(np.power(target-predicted, 2))/2

def mse_prime(target, predicted):
	assert(target.shape == predicted.shape)

	return predicted-target

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_prime(x):
	s = sigmoid(x)
	return s*(1-s)

def relu(x):
	result = np.copy(x)
	result[result<0] = 0
	return result

def relu_prime(x):
	result = np.copy(x)
	result[result<0] = 0
	result[result>0] = 1
	return result

n_samples = X.shape[0]

Z = np.empty(n_samples, dtype=object)
A = np.empty(n_samples, dtype=object)

NB = np.empty(n_samples, dtype=object)
NW = np.empty(n_samples, dtype=object)

ANB = []
ANW = []

Y = produce_target_values([L.shape[0], L_unique.shape[0]], L, L_unique)

act       = sigmoid
act_prime = sigmoid_prime

cst       = mse
cst_prime = mse_prime

batch_size = 23
batch_idx  = 0
batch      = 0

eta = 1

while batch_idx+batch_size<=n_samples:
	
	anb = np.array([np.zeros(b.shape) for b in biases],  dtype=object)
	anw = np.array([np.zeros(w.shape) for w in weights], dtype=object)

	error = 0.0

	for sample_idx in range(batch_size):
		idx = batch_idx+sample_idx

		x = X[idx]

		zis = np.empty(n_layers, dtype=object)
		ais = np.empty(n_layers, dtype=object)
		
		zis[0] = np.copy(x)
		ais[0] = np.copy(x)

		for i, (w, b) in enumerate(zip(weights, biases)):
			zis[i+1] = np.dot(w, ais[i])+b
			ais[i+1] = act(zis[i+1])

		Z[idx] = zis
		A[idx] = ais

		y = Y[idx]

		error += cst(y, ais[-1])

		nb = np.empty(biases.shape[0], dtype=object)
		nw = np.empty(weights.shape[0], dtype=object)

		d = cst_prime(y, ais[-1])*act_prime(zis[-1])
		
		nb[-1] = d
		nw[-1] = np.outer(d, ais[-2].T)

		for l in range(2, n_layers):
			z = zis[-l]
			cp = act_prime(z)

			d = np.dot(weights[-l+1].T, d)*cp

			nb[-l] = d
			nw[-l] = np.outer(d, ais[-l-1].T)

		NB[idx] = nb
		NW[idx] = nw

		for accum_b, nabla_b in zip(anb, nb): accum_b += nabla_b
		for accum_w, nabla_w in zip(anw, nw): accum_w += nabla_w

	ANB.append(anb)
	ANW.append(anw)

	for b, accum_b in zip(biases, anb):  b -= eta/batch_size*accum_b
	for w, accum_w in zip(weights, anw): w -= eta/batch_size*accum_w

	print('Batch {} error: {}'.format(batch, error/batch_size))

	batch_idx += batch_size
	batch     += 1

n_left = n_samples-batch_idx

while batch_idx<n_samples:

	batch_idx += 1