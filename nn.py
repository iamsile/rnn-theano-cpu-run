#pylint: skip-file
import numpy as np
from gru_layer import *
from softmax_layer import *
from lstm import *

class Layer(object):
    def __init__(self, shape):
        pass

    def active(self, X):
        pass

    def calculate_delta(self, propagation = None, Y = None):
        pass

    def update(self, X, lr):
        pass
 
class NN(object):
    def __init__(self, in_size, out_size, hidden_size, cell, p = 0.5):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.p = p
        self.batch_size = T.iscalar('batch_size')
        self.define_layers()

    def define_layers(self):
        self.layers = []
        self.params = []
        rng = np.random.RandomState(1234)

        for layer in xrange(len(self.hidden_size)):
            if layer == 0:
                shape = (self.in_size, self.hidden_size[layer])
            else:
                shape = (self.hidden_size[layer - 1], self.hidden_size[layer])
            if self.cell == "gru":
                hidden_layer = GRULayer(rng, str(layer), shape, self.p)
            elif self.cell == "lstm":
                hidden_layer = LSTMLayer(rng, str(layer), shape, self.p)
            self.layers.append(hidden_layer)
            self.params += hidden_layer.cell.params

        output_layer = SoftmaxLayer((self.hidden_size[len(self.hidden_size) - 1], self.out_size))
        self.layers.append(output_layer)
        self.params += output_layer.cell.params

    def batch_train(self, X, Y, lr):
        self.feed_forward(X)
        self.back_propagarion(Y)
        self.update_parameters(X, lr)

    def feed_forward(self, X):
        for i in xrange(len(self.layers)):
            if i == 0:
                self.layers[i].active(X)
            else:
                self.layers[i].active(self.layers[i - 1].activation)
    
    def back_propagarion(self, Y):
        for i in xrange(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].calculate_delta(None, Y)
            else:
                self.layers[i].calculate_delta(self.layers[i + 1].propagation, None)
    
    def update_parameters(self, X, lr):
        for i in xrange(len(self.layers)):
            if i == 0:
                self.layers[i].update(X, lr)
            else:
                self.layers[i].update(self.layers[i - 1].activation, lr)
    
    def output(self):
        return self.layers[len(self.layers) - 1].activation

    def predict(self, x):
        self.feed_forward(x)
        return self.output()

