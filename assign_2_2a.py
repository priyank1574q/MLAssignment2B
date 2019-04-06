import numpy as np
import pandas as pd
import string
import math
import sys
import os

train_read = sys.argv[1]
train_path = os.path.abspath(train_read)
path_train = os.path.dirname(train_path)
os.chdir(path_train)

test_read = sys.argv[2]
test_path = os.path.abspath(test_read)
path_test = os.path.dirname(test_path)
os.chdir(path_test)

x_train = (pd.read_csv(train_read, header = None, na_filter = False, low_memory = False)).values
x_test = (pd.read_csv(test_read, header = None, na_filter = False, low_memory = False)).values
y_train = list(x_train[:,0])
y_test = list(x_test[:,0])
x_train = x_train[:,1:]
x_test = x_test[:,1:]

vocab = {}
for i in range(x_train.shape[0]):
    #l = (x_train[i][0]).replace('&quot;',' ')
    x_train[i][0] = x_train[i][0].split(' ')
    #x_train[i][0] = (''.join([j.lower() for j in l if j in string.ascii_letters + ' '])).split(" ")
    for j in range(len(x_train[i][0])):
        vocab.update({x_train[i][0][j]: 0})

for i in range(x_test.shape[0]):
    #l = (x_test[i][0]).replace('&quot;',' ')
    x_test[i][0] = x_test[i][0].split(' ')
    #x_test[i][0] = (''.join([j.lower() for j in l if j in string.ascii_letters + ' '])).split(" ")

vocab = sorted(list(vocab))

calc = {}
for i in range(len(vocab)):
    calc.update({vocab[i]: i})

s = np.full((5,len(calc)), 1.0)

for i in range(len(y_train)):
    for j in range(len(x_train[i][0])):
        if x_train[i][0][j] in calc.keys():
            if(y_train[i] == 1):
                s[0, calc[x_train[i][0][j]]] += 1
            elif(y_train[i] == 2):
                s[1, calc[x_train[i][0][j]]] += 1
            elif(y_train[i] == 3):
                s[2, calc[x_train[i][0][j]]] += 1
            elif(y_train[i] == 4):
                s[3, calc[x_train[i][0][j]]] += 1
            elif(y_train[i] == 5):
                s[4, calc[x_train[i][0][j]]] += 1

n = [0]*5
for i in range(s.shape[1]):
    n[0] += s[0, i]
    n[1] += s[1, i]
    n[2] += s[2, i]
    n[3] += s[3, i]
    n[4] += s[4, i]

for i in range(s.shape[0]):
    s[i,:] = s[i,:]/n[i]

ip = [0]*5
for i in range(len(y_train)):
    if(y_train[i] == 1):
        ip[0] += 1
    elif(y_train[i] == 2):
        ip[1] += 1
    elif(y_train[i] == 3):
        ip[2] += 1
    elif(y_train[i] == 4):
        ip[3] += 1
    elif(y_train[i] == 5):
        ip[4] += 1

tot = ip[0] + ip[1] + ip[2] + ip[3] + ip[4]

for i in range(len(ip)):
    ip[i] = ip[i]/tot

def predictor(ex):
    si = [0,0,0,0,0]
    
    ex_vocab = list(set(ex))
    
    ex_vocab_new = []
    
    for i in range(len(ex_vocab)):
        if ex_vocab[i] in calc.keys():
            ex_vocab_new.append(ex_vocab[i])
    
    ex_vocab = list(ex_vocab_new)
    ex_freq = [1.0]*len(ex_vocab)
    ex_calc = {}
    
    for i in range(len(ex_vocab)):
        ex_calc.update({ex_vocab[i]: i})
    
    for i in range(len(ex)):
        if ex[i] in ex_calc.keys():
            ex_freq[ex_calc[ex[i]]] += 1
        
    for j in range(len(si)):
        for i in range(len(ex_vocab)):
            si[j] += ex_freq[ex_calc[ex_vocab[i]]]*math.log(s[j, calc[ex_vocab[i]]])
        si[j] += math.log(ip[j])

    return(np.argmax(np.array(si))+1)

ans = []

for i in range(x_test.shape[0]):
    ans.append(predictor(x_test[i][0]))

out_write = sys.argv[3]
path_out = os.path.abspath(out_write)
out_path = os.path.dirname(path_out)
os.chdir(out_path)
np.savetxt(out_write, ans)