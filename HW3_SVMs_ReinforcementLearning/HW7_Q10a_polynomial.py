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
print('data-processed')

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
print("k_fold done")

# Polynomial KERNEL
#with C = 1
print("C = 1")
clf_C1 = SVC(C = 1, kernel = 'poly', degree = 2).fit(wines_X, wines_y)
print("classified for C = 1\n")
start1 = time.time()
precision_C1 = cross_val_score(clf_C1, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'precision_macro')
end1 = time.time()
print("finished precision\n")
polynomialPrecision_C1_time = end1 - start1
mean_precision_C1 = np.mean(precision_C1)

print("mean_precision_C1: ", mean_precision_C1, "\n")
print("Time elapsed: ", polynomialPrecision_C1_time, "\n")

start2 = time.time()
recall_C1 = cross_val_score(clf_C1, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'recall_macro')
end2 = time.time()
print("finished recall\n")
polynomialRecall_C1_time = end2 - start2
mean_recall_C1 = np.mean(recall_C1)

print("mean_recall_C1: ", mean_recall_C1, "\n")
print("Time elapsed: ", polynomialRecall_C1_time, "\n")

start3 = time.time()
f1score_C1 = cross_val_score(clf_C1, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'f1_macro')
end3 = time.time()
print("finished f1 \n")
polynomialF1Score_C1_time = end3 - start3
mean_f1score_C1 = np.mean(f1score_C1)

print("mean_f1score_C1: ", mean_f1score_C1, "\n")
print("Time elapsed: ", polynomialF1Score_C1_time, "\n")

#with C = 10
print("C = 10")
clf_C10 = SVC(C = 10, kernel = 'poly', degree = 2).fit(wines_X, wines_y)
print("classified for C = 10\n")
start4 = time.time()
precision_C10 = cross_val_score(clf_C10, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'precision_macro')
end4 = time.time()
print("finished precision\n")
polynomialPrecision_C10_time = end4 - start4
mean_precision_C10 = np.mean(precision_C10)

print("mean_precision_C10: ", mean_precision_C10, "\n")
print("Time elapsed: ", polynomialPrecision_C10_time, "\n")

start5 = time.time()
recall_C10 = cross_val_score(clf_C10, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'recall_macro')
end5 = time.time()
print("finished recall\n")
polynomialRecall_C10_time = end5 - start5
mean_recall_C10 = np.mean(recall_C10)

print("mean_recall_C10: ", mean_recall_C10, "\n")
print("Time elapsed: ", polynomialRecall_C10_time, "\n")


start6 = time.time()
f1score_C10 = cross_val_score(clf_C10, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'f1_macro')
end6 = time.time()
print("finished f1\n")
polynomialF1Score_C10_time = end6 - start6
mean_f1score_C10 = np.mean(f1score_C10)

print("mean_f1score_C10: ", mean_f1score_C10, "\n")
print("Time elapsed: ", polynomialF1Score_C10_time, "\n")

#with C = 100
# clf_C100 = SVC(C = 100, kernel = 'polynomial', degree = 2).fit(wines_X, wines_y)
# start7 = time.time()
# precision_C100 = cross_val_score(clf_C100, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'precision_macro')
# end7 = time.time()
# polynomialPrecision_C100_time = end7 - start7
# mean_precision_C100 = np.mean(precision_C100)

# print("mean_precision_C100: ", mean_precision_C100, "\n")
# print("Time elapsed: ", polynomialPrecision_C100_time, "\n")

# start8 = time.time()
# recall_C100 = cross_val_score(clf_C100, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'recall_macro')
# end8 = time.time()
# polynomialRecall_C100_time = end8 - start8
# mean_recall_C100 = np.mean(recall_C100)

# print("mean_recall_C100: ", mean_recall_C100, "\n")
# print("Time elapsed: ", polynomialRecall_C100_time, "\n")

# start9 = time.time()
# f1score_C100 = cross_val_score(clf_C100, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'f1_macro')
# end9 = time.time()
# polynomialF1Score_C100_time = end9 - start9
# mean_f1score_C100 = np.mean(f1score_C100)

# print("mean_f1score_C100: ", mean_f1score_C100, "\n")
# print("Time elapsed: ", polynomialF1Score_C100_time, "\n")

# # with C = 1000
# clf_C1000 = SVC(C = 1000, kernel = 'polynomial', degree = 2).fit(wines_X, wines_y)
# start10 = time.time()
# precision_C1000 = cross_val_score(clf_C1000, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'precision_macro')
# end10 = time.time()
# polynomialPrecision_C1000_time = end10 - start10
# mean_precision_C1000 = np.mean(precision_C1000)

# print("mean_precision_C1000: ", mean_precision_C1000, "\n")
# print("Time elapsed: ", polynomialPrecision_C1000_time, "\n")

# start11 = time.time()
# recall_C1000 = cross_val_score(clf_C1000, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'recall_macro')
# end11 = time.time()
# polynomialRecall_C1000_time = end11 - start11
# mean_recall_C1000 = np.mean(recall_C1000)

# print("mean_recall_C1000: ", mean_recall_C1000, "\n")
# print("Time elapsed: ", polynomialRecall_C1000_time, "\n")

# start12 = time.time()
# f1score_C1000 = cross_val_score(clf_C1000, wines_X, wines_y, cv = k_fold, n_jobs = 1, scoring = 'f1_macro')
# end12 = time.time()
# polynomialF1Score_C1000_time = end12 - start12
# mean_f1score_C1000 = np.mean(f1score_C1000)

# print("mean_f1score_C1000: ", mean_f1score_C1000, "\n")
# print("Time elapsed: ", polynomialF1Score_C1000_time, "\n")

 


