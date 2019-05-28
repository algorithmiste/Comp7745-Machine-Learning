#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier 
import time



wines = pd.read_csv("C:/Users/casey/Desktop/Machine Learning (7745)/Comp 7745 Machine Learning/HWs/HW3 C7745/wines.csv")
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

cols = np.asarray(wines.columns).tolist() #deleted cols.
wines = MultiColumnLabelEncoder(columns = cols).fit_transform(wines)

# print(wines)  # ****successful up until this point
wines_X = wines.drop(['quality'], axis = 1)
wines_y = wines['quality']

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# Vanilla Decision Tree Classifier with max depth = 1 

clfDT_md1 = DecisionTreeClassifier(max_depth = 1).fit(wines_X, wines_y)
precision_clfDT_md1 = cross_val_score(clfDT_md1, wines_X, wines_y, scoring = 'precision_macro')
mean_precision_clfDT_md1 = np.mean(precision_clfDT_md1)

recall_clfDT_md1 = cross_val_score(clfDT_md1, wines_X, wines_y, scoring = 'recall_macro')
mean_recall_clfDT_md1 = np.mean(recall_clfDT_md1)

f1score_clfDT_md1 = cross_val_score(clfDT_md1, wines_X, wines_y, scoring = 'f1_macro')
mean_f1score_clfDT_md1 = np.mean(f1score_clfDT_md1)

print("\nVanilla DT Clf with max_depth = 1: \n")
print("Avg. precision: ", mean_precision_clfDT_md1, "\n")
print("Avg. recall: ", mean_recall_clfDT_md1, "\n")
print("Avg. f1 score: ", mean_f1score_clfDT_md1, "\n\n")


# Adaboosted Decision Tree Classifier with max depth = 1

clfADA_md1 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1)).fit(wines_X, wines_y)
precision_clfADA_md1 = cross_val_score(clfADA_md1, wines_X, wines_y, scoring = 'precision_macro')
mean_precision_clfADA_md1 = np.mean(precision_clfADA_md1)


recall_clfADA_md1 = cross_val_score(clfADA_md1, wines_X, wines_y, scoring = 'recall_macro')
mean_recall_clfADA_md1 = np.mean(recall_clfADA_md1)

f1score_clfADA_md1 = cross_val_score(clfADA_md1, wines_X, wines_y, scoring = 'f1_macro')
mean_f1score_clfADA_md1 = np.mean(f1score_clfADA_md1)

print("\nAdaboosted DT Clf with max_depth = 1: \n")
print("Avg. precision: ", mean_precision_clfADA_md1, "\n")
print("Avg. recall: ", mean_recall_clfADA_md1, "\n")
print("Avg. f1 score: ", mean_f1score_clfADA_md1, "\n\n")


# Vanilla Decision Tree Classifier with max depth = 5
clfDT_md5 = DecisionTreeClassifier(max_depth = 5).fit(wines_X, wines_y)
precision_clfDT_md5 = cross_val_score(clfDT_md5, wines_X, wines_y, scoring = 'precision_macro')
mean_precision_clfDT_md5 = np.mean(precision_clfDT_md5)

recall_clfDT_md5 = cross_val_score(clfDT_md5, wines_X, wines_y, scoring = 'recall_macro')
mean_recall_clfDT_md5 = np.mean(recall_clfDT_md5)

f1score_clfDT_md5 = cross_val_score(clfDT_md5, wines_X, wines_y, scoring = 'f1_macro')
mean_f1score_clfDT_md5 = np.mean(f1score_clfDT_md5)

print("\nVanilla DT Clf with max_depth = 5: \n")
print("Avg. precision: ", mean_precision_clfDT_md5, "\n")
print("Avg. recall: ", mean_recall_clfDT_md5, "\n")
print("Avg. f1 score: ", mean_f1score_clfDT_md5, "\n\n")


# Adaboosted Decision Tree Classifier with max depth = 5
clfADA_md5 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 5)).fit(wines_X, wines_y)
precision_clfADA_md5 = cross_val_score(clfADA_md5, wines_X, wines_y, scoring = 'precision_macro')
mean_precision_clfADA_md5 = np.mean(precision_clfADA_md5)

recall_clfADA_md5 = cross_val_score(clfADA_md5, wines_X, wines_y, scoring = 'recall_macro')
mean_recall_clfADA_md5 = np.mean(recall_clfADA_md5)

f1score_clfADA_md5 = cross_val_score(clfADA_md5, wines_X, wines_y, scoring = 'f1_macro')
mean_f1score_clfADA_md5 = np.mean(f1score_clfADA_md5)

print("\nAdaboosted DT Clf with max_depth = 5: \n")
print("Avg. precision: ", mean_precision_clfADA_md5, "\n")
print("Avg. recall: ", mean_recall_clfADA_md5, "\n")
print("Avg. f1 score: ", mean_f1score_clfADA_md5, "\n\n")


# Vanilla Decision Tree Classifier with max depth = 10
clfDT_md10 = DecisionTreeClassifier(max_depth = 10).fit(wines_X, wines_y)
precision_clfDT_md10 = cross_val_score(clfDT_md10, wines_X, wines_y, scoring = 'precision_macro')
mean_precision_clfDT_md10 = np.mean(precision_clfDT_md10)

recall_clfDT_md10 = cross_val_score(clfDT_md10, wines_X, wines_y, scoring = 'recall_macro')
mean_recall_clfDT_md10 = np.mean(recall_clfDT_md10)

f1score_clfDT_md10 = cross_val_score(clfDT_md10, wines_X, wines_y, scoring = 'f1_macro')
mean_f1score_clfDT_md10 = np.mean(f1score_clfDT_md10)

print("Vanilla DT Clf with max_depth = 10: \n")
print("Avg. precision: ", mean_precision_clfDT_md10, "\n")
print("Avg. recall: ", mean_recall_clfDT_md10, "\n")
print("Avg. f1 score: ", mean_f1score_clfDT_md10, "\n\n")


# Adaboosted Decision Tree Classifier with max depth = 10
clfADA_md10 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 10)).fit(wines_X, wines_y)
precision_clfADA_md10 = cross_val_score(clfADA_md10, wines_X, wines_y, scoring = 'precision_macro')
mean_precision_clfADA_md10 = np.mean(precision_clfADA_md10)

recall_clfADA_md10 = cross_val_score(clfADA_md10, wines_X, wines_y, scoring = 'recall_macro')
mean_recall_clfADA_md10 = np.mean(recall_clfADA_md10)

f1score_clfADA_md10 = cross_val_score(clfADA_md10, wines_X, wines_y, scoring = 'f1_macro')
mean_f1score_clfADA_md10 = np.mean(f1score_clfADA_md10)

print("\nAdaboosted DT Clf with max_depth = 10: \n")
print("Avg. precision: ", mean_precision_clfADA_md10, "\n")
print("Avg. recall: ", mean_recall_clfADA_md10, "\n")
print("Avg. f1 score: ", mean_f1score_clfADA_md10, "\n\n")