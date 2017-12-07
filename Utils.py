import numpy as np
import pandas as pd
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

def scv(X, y, model, cv = 3, test_size = 0.5):
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

def load_shopping_data(directory, features=None, label_name=['TripType']):
    data = pd.read_csv(directory)
    label = data[label_name].values.ravel()
    if features is None:
        content = data.drop(label_name, axis=1)
    else:
        content = data[features].values
    return content, label

class MajorityVoteMask:
    model = None
    classes = 0
    number_of_classes = 0
    group_on_column = None
    
    def __init__(self, model, group_on_column):
        self.model = model
        self.group_on_column = group_on_column
    
    def fit(self, X, y):
        # Ignore the visitNumber column
        X = np.delete(X, self.group_on_column, axis=1)
        self.model.fit(X, y)
        self.classes = np.unique(y)
        self.number_of_classes = len(self.classes)
        
    def predict(self, data):
        pred = self.model.predict(data)
        (values, counts) = np.unique(pred, return_counts=True)
        return values[np.argmax(counts)]
    
    def predict_bulk(self, data):
        #TO BE IMPLEMENTED WHEN HAVE TIME
        pass
    
    def score(self, data, label):
        total = 0
        number_correct = 0
        tempData = np.hstack([data, label.reshape(-1,1)])
        df = pd.DataFrame(tempData)
        val = df.groupby(df.columns[self.group_on_column])
        for visitNumber, item in val:
            item = item.values
            pred = self.predict(item[:, :self.group_on_column])
            if item[0,self.group_on_column+1] == pred:
                number_correct+=1
            total+=1
        return float(number_correct)/total