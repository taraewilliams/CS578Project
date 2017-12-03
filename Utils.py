import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

def confusion_matrix(preds, labels, classes):
    label_reindex = classes
    n = len(label_reindex)
    mat = np.zeros([n,n])
    for pred, label in zip(preds, labels):
        l_index = label_reindex.index(label)
        p_index = label_reindex.index(pred)
        mat[l_index, p_index] = mat[l_index, p_index] + 1
    return mat

def roc_curve(label, score, thresh = 100):
    n= score.shape[0]
    fpr_tpr = np.zeros([2, thresh])
    min_score = score.min()
    max_score = score.max()
    threshold = np.linspace(max_score, min_score, thresh)
    total_positives = label.sum()
    total_negatives = n - total_positives
    for (rounds, t) in enumerate(threshold):
        tp = np.logical_and( score > t, label==1 ).sum()
        fp = np.logical_and( score > t, label==0 ).sum()
        fpr_tpr[0, rounds] = fp / float(total_negatives)
        fpr_tpr[1, rounds] = tp / float(total_positives)
        
    return fpr_tpr[0], fpr_tpr[1]

def auc(fpr, tpr):
    auc = 0.0
    for i in range(fpr.shape[0]-1):
        auc += (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i])
    auc *= 0.5
    return auc

def crossValidation(train, label, model, cv):
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

def scv(X, y, model, cv = 3, test_size = 0.5)
    result = np.zeros([cv, 1])
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size)
    index = 0
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        result[index] = model.score(X_test, y_test)
        index += 1
    return result