#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from nn import * 

class Softmax(object):
    def __init__(self, shape, X):
        prefix = "Softmax_"
        self.in_size, self.out_size = shape
        self.W = init_weights(shape, prefix + "W")
        self.b = init_bias(self.out_size, prefix + "b")

        self.gW = init_gradws(shape, prefix + "gW")
        self.gb = init_bias(self.out_size, prefix + "gb")

        D = T.matrices("D")
        self.X = X
        def _active(X):
            return T.nnet.softmax(T.dot(X, self.W) + self.b)
        self.active = theano.function(inputs = [self.X], outputs = _active(self.X))

        def _propagate(D):
            return T.dot(D, self.W.T)
        self.propagate = theano.function(inputs = [D], outputs = _propagate(D))

        x, dy = T.rows("x","dy")
        updates_grad = [(self.gW, self.gW + T.dot(x.T, dy)),
               (self.gb, self.gb + dy)]
        self.grad = theano.function(
            inputs = [x, dy],
            updates = updates_grad
        )

        updates_clear = [
               (self.gW, self.gW * 0),
               (self.gb, self.gb * 0)]
        self.clear_grad = theano.function(
            inputs = [],
            updates = updates_clear
        )

        lr = T.scalar()
        t = T.scalar()
        updates_w = [
               (self.W, self.W - self.gW * lr / t),
               (self.b, self.b - self.gb * lr / t)]
        self.update = theano.function(
            inputs = [lr, t],
            updates = updates_w
        )

        self.params = [self.W, self.b]

class SoftmaxLayer(object):
    def __init__(self, shape, X):
        self.cell = Softmax(shape, X)
        self.activation = []
        self.delta = []
        self.propagation = []

    def active(self, X):
        self.activation = np.asmatrix(self.cell.active(X)) # bugs here

    def calculate_delta(self, propagation = None, Y = None):
        self.delta = self.activation - Y
        self.propagation = np.asmatrix(self.cell.propagate(self.delta))

    def update(self, X, lr):
        self.cell.clear_grad()
        for t in xrange(len(X)):
            self.cell.grad(X[t,], self.delta[t,])
        self.cell.update(lr, len(X));

