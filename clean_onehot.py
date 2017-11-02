import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

df = pd.read_csv('Data/train.csv')
df['DepartmentDescription'] = df['DepartmentDescription'].apply(str)
df.head()


features = df[['VisitNumber', 'Upc', 'ScanCount', 'FinelineNumber']].values
labels = df[['TripType']].values
enco = LabelBinarizer()
weekdays = enco.fit_transform(df[['Weekday']].values)
department = enco.fit_transform(df[['DepartmentDescription']].values)
labels = enco.fit_transform(df['TripType'].values)
features = df[['VisitNumber', 'Upc', 'ScanCount', 'FinelineNumber']].values
features = np.hstack([features, weekdays, department])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2, random_state=random_state)
np.savez_compressed('Data/cleaned.npz', X_train=X_train, X_test=X_test, y_train = y_train, y_test = y_test)