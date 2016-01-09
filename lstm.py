#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from data import char_sequence
from utils_pg import *

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape, name):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01), name)

def init_bias(size, name):
    return theano.shared(floatX(np.zeros((size, 1))), name) 

class LSTMLayer(object):
    def __init__(self, layer, rng, shape, p):
        prefix = "LSTM_"
        self.in_size, self.out_size = shape
        
        self.W_xi = init_weights((self.in_size, self.out_size), prefix + "W_xi" + "_" + layer)
        self.W_hi = init_weights((self.out_size, self.out_size), prefix + "W_hi" + "_" + layer)
        self.W_ci = init_weights((self.out_size, self.out_size), prefix + "W_ci" + "_" + layer)
        self.b_i = init_bias(self.out_size, prefix + "b_i" + "_" + layer)
        
        self.W_xf = init_weights((self.in_size, self.out_size), prefix + "W_xf" + "_" + layer)
        self.W_hf = init_weights((self.out_size, self.out_size), prefix + "W_hf" + "_" + layer)
        self.W_cf = init_weights((self.out_size, self.out_size), prefix + "W_cf" + "_" + layer)
        self.b_f = init_bias(self.out_size, prefix + "b_f" + "_" + layer)

        self.W_xc = init_weights((self.in_size, self.out_size), prefix + "W_xc" + "_" + layer)
        self.W_hc = init_weights((self.out_size, self.out_size), prefix + "W_hc" + "_" + layer)
        self.b_c = init_bias(self.out_size, prefix + "b_c" + "_" + layer)

        self.W_xo = init_weights((self.in_size, self.out_size), prefix + "W_xo" + "_" + layer)
        self.W_ho = init_weights((self.out_size, self.out_size), prefix + "W_ho" + "_" + layer)
        self.W_co = init_weights((self.out_size, self.out_size), prefix + "W_co" + "_" + layer)
        self.b_o = init_bias(self.out_size, prefix + "b_o" + "_" + layer)

        # self.X = X
        
        def _active(x, pre_h, pre_c):
            i = T.nnet.sigmoid(T.dot(x, self.W_xi) + T.dot(pre_h, self.W_hi) + T.dot(pre_c, self.W_ci) + self.b_i)
            f = T.nnet.sigmoid(T.dot(x, self.W_xf) + T.dot(pre_h, self.W_hf) + T.dot(pre_c, self.W_cf) + self.b_f)
            gc = T.tanh(T.dot(x, self.W_xc) + T.dot(pre_h, self.W_hc) + self.b_c)
            c = f * pre_c + i * gc
            o = T.nnet.sigmoid(T.dot(x, self.W_xo) + T.dot(pre_h, self.W_ho) + T.dot(c, self.W_co) + self.b_o)
            h = o * T.tanh(c)
            return h, c
        [h, c], updates = theano.scan(_active, sequences = [self.X],
                                      outputs_info = [T.alloc(floatX(0.), 1, self.out_size),
                                                      T.alloc(floatX(0.), 1, self.out_size)])
        
        # self.activation = T.reshape(h, (self.X.shape[0], self.out_size))
        h = T.reshape(h, (self.X.shape[0], self.out_size))
        # dropout
        if p > 0:
            srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
            mask = srng.binomial(n = 1, p = 1-p, size = h.shape, dtype = theano.config.floatX)
            # self.activation = T.switch(T.eq(is_train, 1), h * mask, h * (1 - p))
            self.activation = T.switch(T.eq(1, 1), h * mask, h * (1 - p))
        else:
            # self.activation = T.switch(T.eq(is_train, 1), h, h)
            self.activation = T.switch(T.eq(1, 1), h, h)
        
        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                       self.W_xf, self.W_hf, self.W_cf, self.b_f,
                       self.W_xc, self.W_hc,            self.b_c,
                       self.W_xo, self.W_ho, self.W_co, self.b_o]
