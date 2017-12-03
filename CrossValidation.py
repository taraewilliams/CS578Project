
# coding: utf-8

# In[ ]:

import numpy as np
from sklearn.utils import shuffle
# np.random.seed(0)


# In[1]:

def crossValidation(model, cv, train, label):
    (n, d) = train.shape
    result = np.zeros([cv, 1])
    if cv > n:
        print("Cannot have more folds than input data size")
        return
    train, label = shuffle(train, label)
    fold_size = int(np.floor(n/cv))
    for i in range(cv):
        start = i*fold_size
        end = i*fold_size + fold_size
        if i == cv-1:
            end = n
        cur_train = np.vstack([train[:start,:], train[end:,:]]) 
        cur_label = np.hstack([label[:start], label[end:]]) 
        cur_test = train[start:end]
        cur_test_label = label[start:end]
        model.fit(cur_train, cur_label)
        result[i] = model.score(cur_test, cur_test_label)
    return result


# In[ ]:



