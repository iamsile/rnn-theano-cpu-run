#-*- coding: utf-8 -*-
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from nn import *
import data

#theano.config.optimizer = "fast_compile"
#theano.config.exception_verbosity = "high"
seqs, i2w, w2i, data_xy = data.char_sequence("result.txt")

e = 0.01
lr = 0.01
batch_size = 100
hidden_size = [100, 100]

dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

cell = "gru"
print "building..."
model = NN(dim_x, dim_y, hidden_size, cell, p = 0.3) # cell = "gru" or "lstm"

print "training..."
start = time.time()
# for i in xrange(100): # ori data
#     error = 0.0;
#     in_start = time.time()
#     for s in xrange(len(seqs)):
#         seq = seqs[s]
#         X = seq[0 : len(seq) - 1, ] 
#         Y = seq[1 : len(seq), ]
#         model.batch_train(X, Y, lr)
#     in_time = time.time() - in_start

#     error /= len(seqs);
#     if error <= e:
#         break
   
#     print "Iter = " + str(i) + ", Error = " + str(error / len(seqs)) + ", Time = " + str(in_time)
# print time.time() - start
for i in xrange(100):
    for batch_id, xy in data_xy.items():
        # print xy[0]
        X, Y, zero_m, seqs_len = data.index2seqs(seqs, xy[0], w2i)
        #print len(X), len(Y)

        concat_X = np.zeros((seqs_len, len(X) * len(w2i)), dtype = theano.config.floatX)
        concat_Y = concat_X.copy()

        for b_i in xrange(len(X)):
            iX = X[b_i]
            iY = Y[b_i]
            if len(iX) == 0 and len(iY) == 0:
                continue
            #print iX, len(iX)
            #print iY, len(iY)
            for r in xrange(seqs_len - iX.shape[0]):
                iX = np.concatenate((iX, zero_m), axis=0)
                iY = np.concatenate((iY, zero_m), axis=0)

            model.batch_train(iX, iY, lr)

print "save model..."
save_model("rnn.model", model) # it's wrong here

print "load model..."
loaded_model = NN(dim_x, dim_y, hidden_size, cell, p = 0.5)
loaded_model = load_model("/Users/taowei/Documents/rnn-theano-cpu/rnn.model", loaded_model)

X = np.zeros((1, dim_x), np.float32)
a = u"是"
X[0, w2i[a]] = 1
print a,
for i in xrange(6):
    # Y = model.predict(X)
    Y = model.predict(X)#
    Y = Y[Y.shape[0] - 1,:]
    p_label = np.argmax(Y)
    print i2w[p_label],
    # print X.shape
    # print Y.shape
    X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)
    #X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)