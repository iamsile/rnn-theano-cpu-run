# __author__ = 'taowei'
#-*- coding: utf-8 -*-
import time
from rnn import *
import data

# seqs, i2w, w2i = data.char_sequence("./data/toy.txt")
seqs, i2w, w2i, data_xy = data.load_hlm("./data/toy.txt", 50)

lr = 0.5

# layers = []
hidden_size = [100, 100, 100]

cell = "gru"

dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

# for lay in xrange(len(hidden_size)):
#     if lay == 0:
#         shape = (dim_x, hidden_size[lay])
#     else:
#         shape = (hidden_size[lay - 1], hidden_size[lay])
#     layers.append(GRULayer(shape))
# layers.append(SoftmaxLayer((hidden_size[len(hidden_size) - 1], dim_y)))

model = RNN(dim_x, dim_y, hidden_size, cell)
model = load_model("./model/rnn.model", model)

#test
# start = time.time()
# for i in xrange(100):
#     acc = 0.0
#     in_start = time.time()
#     for s in xrange(len(seqs)):
#         seq = seqs[s]
#         X = seq[0 : len(seq) - 1, ]
#         Y = seq[1 : len(seq), ]
#         model.batch_train(X, Y, lr)
#     in_time = time.time() - in_start
#
#     # num_x = 0.0
#     # for s in xrange(len(seqs)):
#     #     seq = seqs[s]
#     #     X = seq[0 : len(seq) - 1, ]
#     #     Y = seq[1 : len(seq), ]
#     #
#     #     label = np.argmax(Y, axis=1)
#     #     p_label = np.argmax(model.predict(X), axis=1)
#     #     print i2w[np.argmax(X[0,])],
#     #     for c in xrange(len(label)):
#     #         num_x += 1
#     #         if label[c] == p_label[c]:
#     #             acc += 1
#     #         print i2w[p_label[c][0,0]],
#     #     print "\n",
#     # print i, acc / num_x, in_time
# print time.time() - start

# X = np.zeros((1, dim_x), np.float32)
# s = u"你"
# X[0, w2i[s]] = 1
# for i in xrange(2):
#     Y = np.argmax(model.predict(X), axis=1)
#     print Y
#     # Y = Y[Y.shape[0] - 1, :]
#     # label = np.argmax(Y)
#     # print i2w[label]
#     # X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)
# print "\n"

num_x = 0.0
acc = 0.0
# X = np.zeros((1, dim_x), np.float32)
#print X
#print len(X)
s = u"谁"
# X[0:w2i[s] - 1,] = 1


X = np.zeros((1, dim_x), np.float32)
a = "嘛"
X[0, w2i[a]] = 1
print a,
for i in xrange(100):
    Y = model.predict(X, 1)
    Y = Y[Y.shape[0] - 1,:]
    p_label = np.argmax(Y)
    print i2w[p_label],
    print X.shape
    print Y.shape
    X = np.concatenate((X, Y), axis=0)
