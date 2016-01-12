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

reload(sys)
sys.setdefaultencoding('utf8')

theano.config.optimizer = "fast_compile"
theano.config.exception_verbosity = "high"
seqs, i2w, w2i, data_xy = data.char_sequence("./data/computer.txt")

e = 0.01
lr = 0.1
drop_rate = 0.3
batch_size = 1
hidden_size = [400, 400]

dim_x = len(w2i)
dim_y = len(w2i)
print dim_x, dim_y

cell = "gru"
print "building..."
model = RNN(dim_x, dim_y, hidden_size, cell, drop_rate) # cell = "gru" or "lstm"

start = time.time()

print "load model..."
loaded_model = RNN(dim_x, dim_y, hidden_size, cell, drop_rate)
loaded_model = load_model("./model/rnn.model", loaded_model)
print loaded_model

fin = open("/Users/taowei/Documents/compter-test.txt")
sysmol = [u"，", u"。", u"！", u"：", u"？", u"~", u"、", u" ?", u"？", u" ", u"、", u"?", u"）", u"（", u"(", u")"]
fout = open("/Users/taowei/Documents/computer-test-out.txt", 'w')
stopword = [u"我", u"你", u"吗", u"啊", u"_", u"；"]

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
    for i in xrange(6):
        Y = loaded_model.predict(X)
        Y = Y[Y.shape[0] - 1, :]
        label = np.argmax(Y)
        if i2w[label] in stopword:
            X = np.concatenate((X, Y), axis = 0)
            i += 1
            continue
        if last != i2w[label]:
            last = i2w[label]
        else:
            X = np.concatenate((X, Y), axis = 0)
            i += 1
            continue
        if i2w[label] not in result:
            result += " " + i2w[label]
        X = np.concatenate((X, Y), axis = 0)
    print "ques" + "\t" + s
    print result
    fout.write("ques" + "\t" + s + "\n")
    fout.write(result.encode("utf-8") + "\n") # windows
    # fout.write(result + "\n")
fin.close()
fout.close()


