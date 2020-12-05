import numpy as np
import pandas as pd

d = pd.read_csv('data/tests_03_12_20.csv', sep=';', decimal=',', encoding='cp1251', header=None, skiprows=4, nrows=460)

dd = d.iloc[:, 2:]
d_norm = (dd-dd.min())/(dd.max()-dd.min())

X = d_norm.iloc[:, 2:].to_numpy(dtype=np.float64)
L = d.iloc[:, 0 ].to_numpy()

L_unique = np.unique(L)

model_shape = [X.shape[1], 16, L_unique.shape[0]]

def init_weights_and_biases(shape):
	assert(len(shape)>1)

	weights = np.array([np.random.randn(shape[i+1], shape[i]) for i in range(len(shape)-1)], dtype=object)
	biases  = np.array([np.zeros(shape[i]) for i in range(1, len(shape))], dtype=object)
	return weights, biases

weights, biases = init_weights_and_biases(model_shape)

def relu(x):
	x[x<0] = 0
	return x

def fwd_once(inputs, weights, biases, act=relu):
	result = np.dot(weights, inputs)+biases
	result = act(result)
	return result

def fwd(inputs, weights, biases, act=relu):
	neurons = np.empty(inputs.shape[0], dtype=object)

	for i, x in enumerate(inputs):
		layers = np.empty(weights.shape[0]+1, dtype=object)

		layers[0] = np.copy(x)
		for j, (w, b) in enumerate(zip(weights, biases)):
			layers[j+1] = fwd_once(layers[j], w, b, act)

		neurons[i] = layers

	return neurons

neurons = fwd(X, weights, biases)

input_neurons  = neurons[0][0]
hidden_neurons = neurons[0][1]
output_neurons = neurons[0][2]
# print(input_neurons, hidden_neurons, output_neurons)

def produce_target_values(shape, labels, labels_set):
	assert(shape[0] == labels.shape[0])

	target = np.zeros(shape)
	for i in range(shape[0]):
		idx = np.where(labels_set == labels[i])
		target[i][idx] = 1.0

	return target

def sel(target, predicted):
	assert(target.shape == predicted.shape)

	return np.sum(np.power(target-predicted, 2))/target.shape[0]

Y = produce_target_values([L.shape[0], L_unique.shape[0]], L, L_unique)

print(output_neurons, Y[0], sel(Y[0], output_neurons))