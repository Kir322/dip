from typing import NamedTuple
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loss functions

class mse:
	def loss(self, x, y):
		return np.sum((y-x)**2)/2

	def grad(self, x, y):
		return x-y

# Layers

class layer:
	def __init__(self):
		self.p = {}
		self.g = {}

class linear(layer):
	def __init__(self, isize, osize):
		super().__init__()

		self.p['w'] = np.random.randn(isize, osize)
		self.p['b'] = np.random.randn(osize)

	def fwd(self, x):
		self.x = x
		return x@self.p['w']+self.p['b']

	def bck(self, g):
		self.g['w'] = self.x.T@g
		self.g['b'] = np.sum(g, axis=0)
		return g@self.p['w'].T

class activation(layer):
	def __init__(self, f, fp):
		super().__init__()

		self.f  = f
		self.fp = fp

	def fwd(self, x):
		self.x = x
		return self.f(x)

	def bck(self, g):
		return self.fp(self.x)*g

def fsigmoid(x):
	return 1/(1+np.exp(-x))

def fsigmoid_prime(x):
	s = fsigmoid(x)
	return s*(1-s)

class sigmoid(activation):
	def __init__(self):
		super().__init__(fsigmoid, fsigmoid_prime)

def ftanh(x):
	return np.tanh(x)

def ftanh_prime(x):
	return 1-ftanh(x)**2

class tanh(activation):
	def __init__(self):
		super().__init__(ftanh, ftanh_prime)

def frelu(x):
	r = np.copy(x)
	r[r<0] = 0
	return r

def frelu_prime(x):
	r = frelu(x)
	r[x>0] = 1
	return r

class relu(activation):
	def __init__(self):
		super().__init__(frelu, frelu_prime)

# Net

class nn:
	def __init__(self, L):
		self.L = L

	def fwd(self, x):
		for l in self.L:
			x = l.fwd(x)
		return x

	def bck(self, g):
		for l in reversed(self.L):
			g = l.bck(g)
		return g

	def png(self):
		for l in self.L:
			for n, p in l.p.items():
				yield p, l.g[n]

# Optimization

class sgd:
	def __init__(self, lr=0.01):
		self.lr = lr

	def step(self, N):
		for p, g in N.png():
			p -= self.lr*g

# Data

Batch = NamedTuple('Batch', [('x', np.ndarray), ('y', np.ndarray)])

def batches(x, y, n=32, shuffle=True):
	assert(len(x)==len(y))
	ss = np.arange(0, len(x), n)
	if shuffle: np.random.shuffle(ss)

	for s in ss:
		yield Batch(x[s:s+n], y[s:s+n])

# Training

def train(N, x, y, nepochs=5000, nbatches=32, loss=mse(), optim=sgd()):
	for e in range(nepochs):
		l = 0.0
		for b in batches(x, y, n=nbatches):
			p  = N.fwd(b.x)
			l += loss.loss(p, b.y)
			g  = loss.grad(p, b.y)

			N.bck(g)
			optim.step(N)

		print('Epoch {} copmleted Loss is {}'.format(e, l/nbatches))

files = glob.glob('data/tests_*.csv')
d = pd.concat([pd.read_csv(f, sep=';', decimal=',', encoding='cp1251', header=None, skiprows=4, nrows=460) for f in files])

dd = d.iloc[:, 2:]
d_norm = (dd-dd.min())/(dd.max()-dd.min())

X = d.iloc[:, 2:].to_numpy(dtype=np.float64)
L = d.iloc[:, 0 ].to_numpy()

L_unique = np.unique(L)

def produce_target_values(shape, labels, labels_set):
	assert(shape[0] == labels.shape[0])

	target = np.zeros(shape)
	for i in range(shape[0]):
		idx = np.where(labels_set == labels[i])
		target[i][idx] = 1.0

	return target

Y = produce_target_values([L.shape[0], L_unique.shape[0]], L, L_unique)

N = nn([linear(isize=X.shape[1], osize=64), relu(),
	    linear(isize=64, osize=L_unique.shape[0]), relu()])

train(N, X, Y, nepochs=5000, nbatches=32, optim=sgd(lr=0.01))

n_guessed_right = 0

for x, y in zip(X, Y):
	p = N.fwd(x)
	
	if np.argmax(p)==np.argmax(y): n_guessed_right+=1
	
print(n_guessed_right, X.shape[0])