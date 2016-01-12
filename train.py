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

model = RNN(dim_x, dim_y, hidden_size, cell)

#batch train
start = time.time()
for i in xrange(100):
    acc = 0.0
    in_start = time.time()
    # for s in xrange(len(seqs)):
    #     seq = seqs[s]
    #     X = seq[0 : len(seq) - 1, ]
    #     Y = seq[1 : len(seq), ]
    #     model.batch_train(X, Y, lr)
    # in_time = time.time() - in_start

    for batch_id, xy in data_xy.items():
        # print xy[0]
        X, Y, zero_m, seqs_len = data.index2seqs(seqs, xy[0], w2i)
        print len(X), len(Y)

        concat_X = np.zeros((seqs_len, len(X) * len(w2i)), dtype = theano.config.floatX)
        concat_Y = concat_X.copy()


        for b_i in xrange(len(X)):
            iX = X[b_i]
            iY = Y[b_i]
            for r in xrange(seqs_len - iX.shape[0]):
                iX = np.concatenate((iX, zero_m), axis=0)
                iY = np.concatenate((iY, zero_m), axis=0)
            model.batch_train(iX, iY, lr)



        # print X, Y
        # model.batch_train(X, Y, lr)
        # label = np.argmax(Y, axis=1)
        # p_label = np.argmax(model.predict(X), axis=1)
        # num = 0.0
        # for c in xrange(len(label)):
        #     num += 1
        #     print i2w[p_label[c][0,0]],

    # num_x = 0.0
    # for s in xrange(len(seqs)):
    #     seq = seqs[s]
    #     X = seq[0 : len(seq) - 1, ]
    #     Y = seq[1 : len(seq), ]
    #
    #     label = np.argmax(Y, axis=1)
    #     p_label = np.argmax(model.predict(X), axis=1)
    #     print i2w[np.argmax(X[0,])],
    #     for c in xrange(len(label)):
    #         num_x += 1
    #         if label[c] == p_label[c]:
    #             acc += 1
    #         print i2w[p_label[c][0,0]],
    #     print "\n",
    # print i, acc / num_x, in_time
print time.time() - start
save_model("./model/rnn.model", model)