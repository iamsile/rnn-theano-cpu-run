#-*- coding: utf-8 -*-
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils import *
from rnn import *
import data
import jieba

theano.config.optimizer = "fast_compile"
theano.config.exception_verbosity = "high"
seqs, i2w, w2i, data_xy = data.char_sequence("./data/computer.txt")

e = 0.01
lr = 0.1
drop_rate = 0.4
batch_size = 1
hidden_size = [400, 400]

dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

cell = "gru"
print "building..."
model = RNN(dim_x, dim_y, hidden_size, cell, p = 0.3) # cell = "gru" or "lstm"

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
# for i in xrange(100):
#     for batch_id, xy in data_xy.items():
#         # print xy[0]
#         X, Y, zero_m, seqs_len = data.index2seqs(seqs, xy[0], w2i)
#         #print len(X), len(Y)

#         concat_X = np.zeros((seqs_len, len(X) * len(w2i)), dtype = theano.config.floatX)
#         concat_Y = concat_X.copy()

#         for b_i in xrange(len(X)):
#             iX = X[b_i]
#             iY = Y[b_i]
#             if len(iX) == 0 and len(iY) == 0:
#                 continue
#             #print iX, len(iX)
#             #print iY, len(iY)
#             for r in xrange(seqs_len - iX.shape[0]):
#                 iX = np.concatenate((iX, zero_m), axis=0)
#                 iY = np.concatenate((iY, zero_m), axis=0)

#             model.batch_train(iX, iY, lr)
# print time.time() - start
# print "save model..."
# save_model("./model/rnn.model", model) # it's wrong here

print "load model..."
loaded_model = RNN(dim_x, dim_y, hidden_size, cell, p = 0.3)
loaded_model = load_model("./model/rnn.model", loaded_model)
print loaded_model

# X = np.zeros((1, dim_x), np.float32)
# a = u"是"
# X[0, w2i[a]] = 1
# print a,
# for i in xrange(6):
#     # Y = model.predict(X)
#     Y = model.predict(X)#
#     Y = Y[Y.shape[0] - 1,:]
#     p_label = np.argmax(Y)
#     print i2w[p_label],
#     # print X.shape
#     # print Y.shape
#     X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)
#     #X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)

fin = open("./data/ques-computer.txt")
sysmol = [u"，", u"。", u"！", u"：", u"？", u"~", u"、", u" ?", u"？", u" ", u"、", u"?", u"）", u"（", u"(", u")"]

for line in fin:
    s = line.strip().lower()
    X = np.zeros((1, dim_x), np.float32)
    res = jieba.cut(s)
    for ss in res:
        if ss in sysmol:
            continue
        if ss not in w2i:
            continue
        X[0, w2i[ss]] = 1

    result = u""
    last = ""
    # print "X.shape: ", X.shape
    for i in xrange(6):
        Y = loaded_model.predict(X)
        # print Y
        # print Y.shape[0]
        # print Y.shape[0] - 1
        Y = Y[Y.shape[0] - 1, :]
        # print Y
        label = np.argmax(Y)
        # if label not in i2w[label]:
        #     print label
        #     continue
        if last != i2w[label]:
            last = i2w[label]
        else:
            # X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)
            X = np.concatenate((X, Y), axis = 0)
            i += 1
            continue
        result += " " + i2w[label]
        # print len(Y)
        # print np.reshape(Y, (1, len(Y)))
        # print np.reshape(Y, (1, len(Y))).reshape
        # X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)
        X = np.concatenate((X, Y), axis = 0)
    print "ques" + "\t" + s
    print result
    # fout.write("ques" + "\t" + s + "\n")
    # fout.write(result.encode("utf-8") + "\n")
fin.close()
# fout.close()


