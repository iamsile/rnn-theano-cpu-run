#-*- coding: utf
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip
import jieba

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def index2seqs(lines, x_index, w2i):
    seqs = []
    for i in x_index:
        line = lines[i]
        x = np.zeros((len(line), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(line)):
            x[j, w2i[line[j]]] = 1
        seqs.append(np.asmatrix(x))

    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    dim = len(w2i)
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
    # print len(seqs)
    for i in xrange(len(seqs)):
        seq = seqs[i]
        X = seq[0 : len(seq) - 1, ]
        Y = seq[1 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])
        batch_y.append(Y)
    return batch_x, batch_y, zeros_m, np.max(seqs_len)

    max_len = np.max(seqs_len);
    mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
    concat_X = np.zeros((max_len, len(batch_x) * dim), dtype = theano.config.floatX)
    concat_Y = concat_X.copy()
    print max_len, len(batch_x) * dim

    for b_i in xrange(len(batch_x)):
        X = batch_x[b_i]
        Y = batch_y[b_i]
        mask[0 : X.shape[0], b_i] = 1
        for r in xrange(max_len - X.shape[0]):
            X = np.concatenate((X, zeros_m), axis=0)
            Y = np.concatenate((Y, zeros_m), axis=0)
        concat_X[:, b_i * dim : (b_i + 1) * dim] = X
        concat_Y[:, b_i * dim : (b_i + 1) * dim] = Y
    # print len(concat_X), len(concat_Y)
    return concat_X, concat_Y, mask, len(batch_x)

def batch_index(seqs, i2w, w2i, batch_size):
    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    for i in xrange(len(seqs)):
        batch_x.append(i)
        batch_y.append(i)
        if len(batch_x) == batch_size or (i == len(seqs) - 1):
            data_xy[batch_id] = [batch_x, batch_y, [], len(batch_x)]
            batch_x = []
            batch_y = []
            batch_id += 1
    return data_xy

def char_sequence(f_path = None, batch_size = 1):
    jieba.load_userdict("./data/dic.txt")
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    data_xy = {}
    sysmol = [u"，", u"。", u"！", u"：", u"？", u"~", u"、", u" ?", u"？", u" ", u"、", u"?", u"）", u"（", u"(", u")" u"!"]
    f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        if len(line) < 3:
            continue
        seg_list = jieba.cut(line)

        w_line = []
        for w in seg_list:
            if w in sysmol:
                continue
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            w_line.append(w)
            if len(w_line) == 100:
                lines.append(w_line)
                w_line = []
        if len(w_line) < 100:
            lines.append(w_line)
    f.close
    seqs = lines
    data_xy = batch_index(seqs, i2w, w2i, batch_size)
    print "#dic = " + str(len(w2i))
    return seqs, i2w, w2i, data_xy

