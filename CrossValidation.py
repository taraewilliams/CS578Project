
# coding: utf-8

# In[1]:

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
# np.random.seed(0)


# In[2]:

def cross_validate(model, data, label, cv = 5, scoring = 'basic'):
    #(n, d) = data.shape
    n = len(data)
    index = np.random.permutation(n)
    k_folds = int(n/cv)
    result = np.zeros([cv,1])
    pred_prob = np.zeros([cv, k_folds])
    for i in range(cv):
        (train, train_label, val, val_label) = cv_split_data(data, label, i, index, k_folds)
        value = cv_score(train, train_label, val, val_label, model, scoring)
        result[i] = value[0]
        pred_prob[i, :] = value[1]
    return result, pred_prob


# In[63]:

def cv_split_data(data, label, i, index, cv):
    val_start = int(i*cv)
    val_end = val_start+cv
    
    train_index = np.hstack((index[:val_start], index[val_end:]))
    train = data[train_index, :]
    label = label[train_index]
    
    val_index = index[val_start:val_end]
    val = data[val_index]
    val_label = label[val_index]
    return (train, label, val, val_label)


# In[54]:

def cv_score(data, label, val, val_label, model, scoring = 'basic'):
    model.fit(data, label)
    if scoring == 'basic':
        pred = model.predict(val)
        return accuracy_score(pred, val_label), model.predict_proba(val)
    #elif scoring == 'average_class':
    #    return mode.score(val, val_label), model.predict_proba(val)
    #else:
    #    return scoring(model, val, val_label)


# In[98]:

def cv_split_data_test():
    data = np.matrix("1 2 3; 4 5 6; 7 8 9; 10 11 12")
    label = np.matrix("1; 1; -1; -1")
    index = np.array([1, 3, 0, 2])
    packet = cv_split_data(data, label, 0, index, 1)
    if packet[0].shape == (3, 3) and packet[1].shape == (3,1) and packet[2].shape == (1, 3) and packet[3].shape == (1,1) and packet[0][0,0] == 10:
            print("Passed 1")
    else:
        print("Failed 1")
    packet = cv_split_data(data, label, 3, index, 3)
    if packet[0].shape == (4, 3) and packet[1].shape == (4,1) and packet[2].shape == (0, 3) and packet[3].shape == (0,1) and packet[0][0,0] == 4:
        print("Passed 2")
    else:
        print("Failed 2")
cv_split_data_test()


# In[ ]:



