import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import jit, njit, types
from numba.experimental import jitclass

# Loss functions

@jitclass
class mse:
	def __init__(self):
		pass

	def loss(self, x, y):
		return np.sum((y-x)**2)/2

	def grad(self, x, y):
		return x-y

class msle:
	def loss(self, x, y):
		return np.sum((y-np.log(x))**2)/2

	def grad(self, x, y):
		return (np.log(x)-y)/x

class mae:
	def loss(self, x, y):
		return np.average(np.abs(x-y))

	def grad(self, x, y):
		r = np.zeros(x.shape)
		r[x>y] =  1
		r[x<y] = -1
		return r

# Layers

#@jitclass(spec=[
#	('w', types.float64[:,:]),
#	('gw', types.float64[:,:]),
#	('b', types.float64[:]),
#	('gb', types.float64[:]),
#	('x', types.float64[:,:])
#])
class linear:
	def __init__(self, isize, osize):
		self.w = np.random.randn(isize, osize)
		self.b = np.random.randn(osize)

	def fwd(self, x):
		self.x = x
		return x@self.w+self.b

	def bck(self, g):
		self.gw = self.x.T@g
		self.gb = np.sum(g, axis=0)
		return g@self.w.T

class activation:
	def __init__(self, f, fp):
		self.w = []
		self.b = []
		self.gw = []
		self.gb = []

		self.f  = f
		self.fp = fp

	def fwd(self, x):
		self.x = x
		return self.f(x)

	def bck(self, g):
		return self.fp(self.x)*g

@njit
def fsigmoid(x):
	return 1/(1+np.exp(-x))

@njit
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
	r[r>0] = 1
	return r

class relu(activation):
	def __init__(self):
		super().__init__(frelu, frelu_prime)

def fsoftplus(x):
	return np.log(1+np.exp(x))

def fsoftplus_prime(x):
	return 1/(1+np.exp(-x))

class softplus(activation):
	def __init__(self):
		super().__init__(fsoftplus, fsoftplus_prime)

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
			if len(l.w)!=0:
				yield l.w, l.gw

			if len(l.b)!=0:
				yield l.b, l.gb

# Optimization

class sgd:
	def __init__(self, lr=0.01):
		self.lr = lr

	def step(self, N):
		for p, g in N.png():
			p -= self.lr*g
			g  = np.zeros(g.shape)

# Data

def batches(x, y, n=32, shuffle=True):
	assert(len(x)==len(y))
	ss = np.arange(0, len(x), n)
	if shuffle: np.random.shuffle(ss)

	for s in ss:
		yield x[s:s+n], y[s:s+n]

# Training

def train(N, X, Y, nepochs=5000, nbatches=32, loss=mse(), optim=sgd()):
	ls = []
	ns = []
	for e in range(nepochs):
		l = 0.0
		n = 0
		for x, y in batches(X, Y, n=nbatches):
			p = N.fwd(x)
			
			xp = np.argmax(p, axis=1)
			yp = np.argmax(y, axis=1)

			n += np.sum(xp==yp)
			
			l += loss.loss(p, y)
			g  = loss.grad(p, y)

			N.bck(g)
			optim.step(N)

		ls.append(l)
		ns.append(n/len(X))

		print('Epoch {} copmleted Loss is {} Correct predictions {}/{} {}'.format(e, l, n, len(X), n/len(X)))

	f, (a1, a2) = plt.subplots(1, 2)

	a1.plot(np.arange(nepochs), ls)
	#a1.xlabel('Epoch')
	#a1.ylabel('Loss')

	a2.plot(np.arange(nepochs), ns)
	#a2.xlabel('Epoch')
	#a2.ylable('Correctness')

	plt.show()

files = glob.glob('data/tests_*.csv')
d = pd.concat([pd.read_csv(f, sep=';', decimal=',', encoding='cp1251', header=None, skiprows=4, nrows=460) for f in files])

dd = d.iloc[:, 2:]
d_norm = (dd-dd.min())/(dd.max()-dd.min())

X = d_norm.iloc[:, 2:].to_numpy(dtype=np.float64)
L = d.iloc[:, 0 ].to_numpy()

L_unique = np.unique(L)

gg = d.groupby(0).sum()

w = 0.4

hh = d[d[0]=='Харківська']
print(hh)

h = hh.groupby(1).sum().reset_index()

print(h)
print(h[1], h[2])

plt.barh(h[1], h[2])
#plt.xticks(rotation=45)
plt.title('Кількість позитивних результатів на COVID-19 У Харковi')
#plt.legend()
plt.show()

assert(False)

plt.bar(L_unique, gg[8], width=0.8, label='Всього')
plt.bar(L_unique, gg[7], width=0.6, label='З пневмоніями')
plt.xticks(rotation=45)
plt.title('Кількість позитивних результатів на COVID-19')
plt.legend()
plt.show()

plt.bar(L_unique, gg[2] , width=0.8, label='Метод ПЛР') # Метод ПЛР
plt.bar(L_unique, gg[18], width=0.6, label='Метод ІФА') # Метод ІФА
plt.bar(L_unique, gg[24], width=0.4, label='Виявлення антигенів') # Виявлення антигенів
plt.xticks(rotation=45)
plt.title('Кількість лабораторно обстежених осіб')
plt.legend()
plt.show()

ls = np.linspace(0.0, 1.0 + w + 0.2, d_norm.shape[0])

plt.subplot(131)
plt.scatter(d[2], ls)
plt.xlabel('Кількість лабораторно обстежених осіб, всього')

plt.subplot(132)
plt.scatter(d[8], ls)
plt.xlabel('Кількість позитивних результатів на COVID-19, всього')

plt.subplot(133)
plt.scatter(d[14], ls)
plt.xlabel('Кількість непротестованих зразків, що залишаються в лабораторії, всього')

plt.show()


assert(False)

def produce_target_values(shape, labels, labels_set):
	assert(shape[0] == labels.shape[0])

	target = np.zeros(shape)
	for i in range(shape[0]):
		idx = np.where(labels_set == labels[i])
		target[i][idx] = 1.0

	return target

Y = produce_target_values([L.shape[0], L_unique.shape[0]], L, L_unique)

lrs = np.linspace(1e-5, 1, num=20)
acs = []

#N = nn([linear(isize=X.shape[1], osize=100), sigmoid(),
#        linear(isize=100, osize=L_unique.shape[0]), sigmoid()])
#
# Learning rate: 0.0526

#N = nn([linear(isize=X.shape[1], osize=100), relu(),
#        linear(isize=100, osize=L_unique.shape[0]), sigmoid()])
#
# Learning rate: 0.0526

#for lr in lrs:
N = nn([linear(isize=X.shape[1], osize=100), relu(),
		#linear(isize=100, osize=100), sigmoid(),
		linear(isize=100, osize=L_unique.shape[0]), sigmoid()])

train(N, X, Y, nepochs=500, nbatches=32, optim=sgd(lr=0.0526), loss=mse())

assert(False)

W = [np.random.randn(X.shape[1], 100), np.random.randn(100, L_unique.shape[0])]
B = [np.random.randn(100), np.random.randn(25)]

@njit
def lin_fwd(x, w, b):
	return x@w+b

MSE      = mse()
lr       = 0.0526
nepochs  = 2000
nbatches = 32

def tt(nepochs=2000, nbatches=32, lr=0.0526):
	ls = []
	ns = []

	for e in range(nepochs):
		l = 0.0
		n = 0
		for x, y in batches(X, Y, n=nbatches):

			z1 = lin_fwd(x, W[0], B[0])
			a1 = frelu(z1)

			z2 = lin_fwd(a1, W[1], B[1])
			a2 = fsigmoid(z2)

			xp = np.argmax(a2, axis=1)
			yp = np.argmax(y , axis=1)

			n += np.sum(xp==yp)

			l += MSE.loss(a2, y)
			g  = MSE.grad(a2, y)

			ga2 = fsigmoid_prime(z2)*g

			gzw2 = a1.T@ga2
			gzb2 = np.sum(ga2, axis=0)
			gz2  = ga2@W[1].T

			ga1 = frelu_prime(z1)*gz2

			gzw1 = x.T@ga1
			gzb1 = np.sum(ga1, axis=0)
			gz1  = ga1@W[0].T

			W[1] -= lr*gzw2
			B[1] -= lr*gzb2

			W[0] -= lr*gzw1
			B[0] -= lr*gzb1

		ls.append(l)
		ns.append(n/len(X))

		print('Epoch {} copmleted Loss is {} Correct predictions {}/{} {}'.format(e, l, n, len(X), n/len(X)))

	f, (a1, a2) = plt.subplots(1, 2)

	a1.plot(np.arange(nepochs), ls)
	#a1.xlabel('Epoch')
	#a1.ylabel('Loss')

	a2.plot(np.arange(nepochs), ns)
	#a2.xlabel('Epoch')
	#a2.ylable('Correctness')

	plt.show()

tt()

n = 0
for x, y in zip(X, Y):
	p = N.fwd(x)
	
	if np.argmax(p)==np.argmax(y): n+=1
	
acs.append(n/len(X))

#plt.plot(lrs, acs)
#plt.show()