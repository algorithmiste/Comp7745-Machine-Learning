import io 
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

'''You will experiment with the following learners.

i) Neural networks (MLPClassifier in sklearn)
ii) Naïve Bayes (MultinomialNB in sklearn)
iii) Logistic Regression (LogisticRegression in sklearn)
iv) AdaBoosting (AdaBoostClassifier in sklearn)
v) SVM (svm.svc in sklearn) '''

'''i. Run 5-fold Cross Validation on the training.txt using the 5 learning algorithms.
Report the average-precision, average-recall and average-F1-scores. You can do this
quite easily using the cross_val_score() function in sklearn
In each algorithm, try to explore different settings of the parameters to achieve best
possible results (this step is largely experimental, try to automate as much as
possible). Parameters that you should try to change include
a. In neural networks change the number of hidden layers and number of units in
each layer
b. In SVMs, change the penalty parameter C and the kernel type
c. In Adaboosting change the number of estimators (n_estimators)
d. In Logistic regression change the penalty: L1 regularization that can also perform
feature selection and L2. Also change the regularization strength parameter ( C ) '''


vectorizer = TfidfVectorizer()
training = open("C:\\Users\\casey\\Desktop\\Project_Comp7745\\project\\training.txt", encoding = 'utf-8')
#split on tab from original testing file - use word_matrix as X and rating Y
#use stopwords in vectorizer.
#pass X into clf as X'

M = pd.read_csv(r"C:\\Users\\casey\\Desktop\\Project_Comp7745\\project\\training.txt", sep='\t', names = ('0', '1'))
X = M.iloc[:,0]

X_matrix = vectorizer.fit_transform(X) #(9994, 23274) 
y = M.iloc[:,1] # (9994,)

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

# # Vanilla AdaBoost, n_estimators = 50
# clf_AB = AdaBoostClassifier().fit(X_matrix, y)
# precision_AB = cross_val_score(clf_AB, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB = np.mean(precision_AB)
# recall_AB = cross_val_score(clf_AB, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB = np.mean(recall_AB)
# f1Score_AB = cross_val_score(clf_AB, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB = np.mean(f1Score_AB)
# print("Mean f1Score with AdaBoosting (No base_estimator), n_estimators = 50: ", mean_f1Score_AB, "\n")

# # Vanilla AdaBoost, n_estimators = 100
# clf_AB = AdaBoostClassifier(n_estimators = 100).fit(X_matrix, y)
# precision_AB = cross_val_score(clf_AB, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB = np.mean(precision_AB)
# recall_AB = cross_val_score(clf_AB, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB = np.mean(recall_AB)
# f1Score_AB = cross_val_score(clf_AB, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB = np.mean(f1Score_AB)
# print("Mean f1Score with AdaBoosting (No base_estimator), n_estimators = 100: ", mean_f1Score_AB, "\n")

#Naive Bayes
# clf_NB = MultinomialNB().fit(X_matrix, y)
# precision_NB = cross_val_score(clf_NB, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_NB = np.mean(precision_NB)
# recall_NB = cross_val_score(clf_NB, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_NB = np.mean(recall_NB)
# f1Score_NB = cross_val_score(clf_NB, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_NB = np.mean(f1Score_NB)
# print("Mean f1Score with Naive Bayes (base): ", mean_f1Score_NB, "\n")

# #AdaBoosted Naive Bayes, n_estimators = 50
# clf_AB_NB = AdaBoostClassifier(base_estimator = MultinomialNB()).fit(X_matrix, y)
# precision_AB_NB = cross_val_score(clf_AB_NB, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_NB = np.mean(precision_AB_NB)
# recall_AB_NB = cross_val_score(clf_AB_NB, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_NB = np.mean(recall_AB_NB)
# f1Score_AB_NB = cross_val_score(clf_AB_NB, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_NB = np.mean(f1Score_AB_NB)
# print("Mean f1Score with AdaBoosted Naive Bayes (base), n_estimators = 50: ", mean_f1Score_AB_NB, "\n")

# #AdaBoosted Naive Bayes, n_estimators = 100
# clf_AB_NB = AdaBoostClassifier(base_estimator = MultinomialNB(), n_estimators = 100).fit(X_matrix, y)
# precision_AB_NB = cross_val_score(clf_AB_NB, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_NB = np.mean(precision_AB_NB)
# recall_AB_NB = cross_val_score(clf_AB_NB, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_NB = np.mean(recall_AB_NB)
# f1Score_AB_NB = cross_val_score(clf_AB_NB, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_NB = np.mean(f1Score_AB_NB)
# print("Mean f1Score with AdaBoosted Naive Bayes (base), n_estimators = 100: ", mean_f1Score_AB_NB, "\n")

# #######################################################################################################

# #Logistic Regression - L2, C = 1
# clf_LR = LogisticRegression().fit(X_matrix, y)
# precision_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_LR = np.mean(precision_LR)
# recall_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_LR = np.mean(recall_LR)
# f1Score_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_LR = np.mean(f1Score_LR)
# print("Mean f1Score with L2 Logistic Regression, C=1 (base): ", mean_f1Score_LR, "\n")

# #AdaBoosted Logistic Regression - L2, C = 1
# clf_AB_LR = AdaBoostClassifier(base_estimator = LogisticRegression()).fit(X_matrix, y)
# precision_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_LR = np.mean(precision_AB_LR)
# recall_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_LR = np.mean(recall_AB_LR)
# f1Score_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_LR = np.mean(f1Score_AB_LR)
# print("Mean f1Score with AdaBoosted L2 Logistic Regression, C=1 (base), n_estimators = 50: ", mean_f1Score_AB_LR, "\n")

# #Logistic Regression with L1, C = 1
# clf_LR = LogisticRegression(penalty = 'l1').fit(X_matrix, y)
# precision_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_LR = np.mean(precision_LR)
# recall_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_LR = np.mean(recall_LR)
# f1Score_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_LR = np.mean(f1Score_LR)
# print("Mean f1Score with L1 Logistic Regression, C=1 (base): ", mean_f1Score_LR, "\n")

# #AdaBoosted Logistic Regression - L1, C = 1
# clf_AB_LR = AdaBoostClassifier(base_estimator = LogisticRegression(penalty = 'l1'), n_estimators = 100).fit(X_matrix, y)
# precision_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_LR = np.mean(precision_AB_LR)
# recall_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_LR = np.mean(recall_AB_LR)
# f1Score_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_LR = np.mean(f1Score_AB_LR)
# print("Mean f1Score with AdaBoosted L1 Logistic Regression, C=1 (base), n_estimators = 100: ", mean_f1Score_AB_LR, "\n")

# ################################################################################

# #Logistic Regression - L2, C = 10
# clf_LR = LogisticRegression(C = 10).fit(X_matrix, y)
# precision_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_LR = np.mean(precision_LR)
# recall_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_LR = np.mean(recall_LR)
# f1Score_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_LR = np.mean(f1Score_LR)
# print("Mean f1Score with L2 Logistic Regression, C=10 (base): ", mean_f1Score_LR, "\n")

# #AdaBoosted Logistic Regression - L2, C = 1
# clf_AB_LR = AdaBoostClassifier(base_estimator = LogisticRegression(C=10)).fit(X_matrix, y)
# precision_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_LR = np.mean(precision_AB_LR)
# recall_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_LR = np.mean(recall_AB_LR)
# f1Score_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_LR = np.mean(f1Score_AB_LR)
# print("Mean f1Score with AdaBoosted L2 Logistic Regression, C=10 (base), n_estimators = 50: ", mean_f1Score_AB_LR, "\n")

# #Logistic Regression with L1, C=10
# clf_LR = LogisticRegression(penalty = 'l1', C=10).fit(X_matrix, y)
# precision_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_LR = np.mean(precision_LR)
# recall_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_LR = np.mean(recall_LR)
# f1Score_LR = cross_val_score(clf_LR, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_LR = np.mean(f1Score_LR)
# print("Mean f1Score with L1 Logistic Regression, C=10 (base): ", mean_f1Score_LR, "\n")

# #AdaBoosted Logistic Regression - L1, C=10
# clf_AB_LR = AdaBoostClassifier(base_estimator = LogisticRegression(penalty = 'l1', C=10), n_estimators = 100).fit(X_matrix, y)
# precision_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_LR = np.mean(precision_AB_LR)
# recall_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_LR = np.mean(recall_AB_LR)
# f1Score_AB_LR = cross_val_score(clf_AB_LR, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_LR = np.mean(f1Score_AB_LR)
# print("Mean f1Score with AdaBoosted L1 Logistic Regression, C=10 (base), n_estimators = 100: ", mean_f1Score_AB_LR, "\n")

# #SVM RBF, C=1
# clf_SVM = SVC().fit(X_matrix, y)
# precision_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_SVM = np.mean(precision_SVM)
# recall_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_SVM = np.mean(recall_SVM)
# f1Score_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_SVM = np.mean(f1Score_SVM)
# print("Mean f1Score with SVM - rbf, C=1: ", mean_f1Score_SVM, "\n")

# #SVM RBF, C=5
# clf_SVM = SVC(C = 5).fit(X_matrix, y)
# precision_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_SVM = np.mean(precision_SVM)
# recall_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_SVM = np.mean(recall_SVM)
# f1Score_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_SVM = np.mean(f1Score_SVM)
# print("Mean f1Score with SVM - rbf, C=5: ", mean_f1Score_SVM, "\n")

#SVM Linear, C=1
# clf_SVM = SVC(kernel = "linear").fit(X_matrix, y)
# precision_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_SVM = np.mean(precision_SVM)
# recall_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_SVM = np.mean(recall_SVM)
# f1Score_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_SVM = np.mean(f1Score_SVM)
# print("Mean f1Score with SVM - linear, C=1: ", mean_f1Score_SVM, "\n")

#AdaBoosted SVM, C = 1
# clf_AB_SVM = AdaBoostClassifier(SVC(probability = True, kernel = "linear")).fit(X_matrix, y)
# precision_AB_SVM = cross_val_score(clf_AB_SVM, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_SVM = np.mean(precision_AB_SVM)
# recall_AB_SVM = cross_val_score(clf_AB_SVM, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_SVM = np.mean(recall_AB_SVM)
# f1Score_AB_SVM = cross_val_score(clf_AB_SVM, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_SVM = np.mean(f1Score_AB_SVM)
# print("Mean f1Score with AdaBoosted SVM - linear, C=1: ", mean_f1Score_AB_SVM, "\n")

# #SVM Linear, C=5
# clf_SVM = SVC(kernel = "linear", C = 5).fit(X_matrix, y)
# precision_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'precision_macro')
# mean_precision_SVM = np.mean(precision_SVM)
# recall_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'recall_macro')
# mean_recall_SVM = np.mean(recall_SVM)
# f1Score_SVM = cross_val_score(clf_SVM, X_matrix, y, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_SVM = np.mean(f1Score_SVM)
# print("Mean f1Score with SVM - linear, C=5: ", mean_f1Score_SVM, "\n")


test = open("C:\\Users\\casey\\Desktop\\Project_Comp7745\\project\\test.txt", encoding = 'utf-8')
#split on tab from original testing file - use word_matrix as X and rating Y
#use stopwords in vectorizer.
#pass X into clf as X'

N = pd.read_csv(r"C:\\Users\\casey\\Desktop\\Project_Comp7745\\project\\test.txt", sep='\t', names = ('0', '1'))
X2 = N.iloc[:,0]

X_matrix2 = vectorizer.fit_transform(X2)  
y2 = N.iloc[:,1] 

#SVM Linear, C=1
# clf_SVM2 = SVC(kernel = "linear").fit(X_matrix2, y2)
# precision_SVM2 = cross_val_score(clf_SVM2, X_matrix2, y2, cv = k_fold, scoring = 'precision_macro')
# mean_precision_SVM2 = np.mean(precision_SVM2)
# print(mean_precision_SVM2)
# recall_SVM2 = cross_val_score(clf_SVM2, X_matrix2, y2, cv = k_fold, scoring = 'recall_macro')
# mean_recall_SVM2 = np.mean(recall_SVM2)
# print(mean_recall_SVM2)
# f1Score_SVM2 = cross_val_score(clf_SVM2, X_matrix2, y2, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_SVM2 = np.mean(f1Score_SVM2)
# print("Mean f1Score with SVM - linear, C=1, on Test data: ", mean_f1Score_SVM2, "\n")

# #Logistic Regression - L2, C = 1
# clf_LR2 = LogisticRegression().fit(X_matrix2, y2)
# precision_LR2 = cross_val_score(clf_LR2, X_matrix2, y2, cv = k_fold, scoring = 'precision_macro')
# mean_precision_LR2 = np.mean(precision_LR2)
# print(mean_precision_LR2)
# recall_LR2 = cross_val_score(clf_LR2, X_matrix2, y2, cv = k_fold, scoring = 'recall_macro')
# mean_recall_LR2 = np.mean(recall_LR2)
# print(mean_recall_LR2)
# f1Score_LR2 = cross_val_score(clf_LR2, X_matrix2, y2, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_LR2 = np.mean(f1Score_LR2)
# print("Mean f1Score with L2 Logistic Regression, C=1, on Test data: ", mean_f1Score_LR2, "\n")

# #AdaBoosted Logistic Regression - L2, C = 1
# clf_AB_LR = AdaBoostClassifier(base_estimator = LogisticRegression(C=1)).fit(X_matrix2, y2)
# precision_AB_LR = cross_val_score(clf_AB_LR, X_matrix2, y2, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_LR = np.mean(precision_AB_LR)
# recall_AB_LR = cross_val_score(clf_AB_LR, X_matrix2, y2, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_LR = np.mean(recall_AB_LR)
# f1Score_AB_LR = cross_val_score(clf_AB_LR, X_matrix2, y2, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_LR = np.mean(f1Score_AB_LR)
# print("Mean f1Score with AdaBoosted L2 Logistic Regression, C=1 (base), n_estimators = 50: ", mean_f1Score_AB_LR, "\n")


# #AdaBoosted Logistic Regression - L1, C = 1
# clf_AB_LR = AdaBoostClassifier(base_estimator = LogisticRegression(penalty = 'l1'), n_estimators = 100).fit(X_matrix2, y2)
# precision_AB_LR = cross_val_score(clf_AB_LR, X_matrix2, y2, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_LR = np.mean(precision_AB_LR)
# recall_AB_LR = cross_val_score(clf_AB_LR, X_matrix2, y2, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_LR = np.mean(recall_AB_LR)
# f1Score_AB_LR = cross_val_score(clf_AB_LR, X_matrix2, y2, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_LR = np.mean(f1Score_AB_LR)
# print("Mean f1Score with AdaBoosted L1 Logistic Regression, C=1 (base), n_estimators = 100: ", mean_f1Score_AB_LR, "\n")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# #AdaBoosted Naive Bayes, n_estimators = 50
# clf_AB_NB = AdaBoostClassifier(base_estimator = MultinomialNB()).fit(X_matrix2, y2)
# precision_AB_NB = cross_val_score(clf_AB_NB, X_matrix2, y2, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB_NB = np.mean(precision_AB_NB)
# recall_AB_NB = cross_val_score(clf_AB_NB, X_matrix2, y2, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB_NB = np.mean(recall_AB_NB)
# f1Score_AB_NB = cross_val_score(clf_AB_NB, X_matrix2, y2, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB_NB = np.mean(f1Score_AB_NB)
# print("Mean f1Score with AdaBoosted Naive Bayes (base), n_estimators = 50: ", mean_f1Score_AB_NB, "\n")

# # Vanilla AdaBoost, n_estimators = 50
# clf_AB = AdaBoostClassifier().fit(X_matrix2, y2)
# precision_AB = cross_val_score(clf_AB, X_matrix2, y2, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB = np.mean(precision_AB)
# recall_AB = cross_val_score(clf_AB, X_matrix2, y2, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB = np.mean(recall_AB)
# f1Score_AB = cross_val_score(clf_AB, X_matrix2, y2, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB = np.mean(f1Score_AB)
# print("Mean f1Score with AdaBoosting (No base_estimator), n_estimators = 50: ", mean_f1Score_AB, "\n")

# # Vanilla AdaBoost, n_estimators = 100
# clf_AB = AdaBoostClassifier(n_estimators = 100).fit(X_matrix2, y2)
# precision_AB = cross_val_score(clf_AB, X_matrix2, y2, cv = k_fold, scoring = 'precision_macro')
# mean_precision_AB = np.mean(precision_AB)
# recall_AB = cross_val_score(clf_AB, X_matrix2, y2, cv = k_fold, scoring = 'recall_macro')
# mean_recall_AB = np.mean(recall_AB)
# f1Score_AB = cross_val_score(clf_AB, X_matrix2, y2, cv = k_fold, scoring = 'f1_macro')
# mean_f1Score_AB = np.mean(f1Score_AB)
# print("Mean f1Score with AdaBoosting (No base_estimator), n_estimators = 100: ", mean_f1Score_AB, "\n")

#SVM Linear, C=1
clf_SVM2 = AdaBoostClassifier(SVC(probability = True, kernel = "linear")).fit(X_matrix2, y2)
precision_SVM2 = cross_val_score(clf_SVM2, X_matrix2, y2, cv = k_fold, scoring = 'precision_macro')
mean_precision_SVM2 = np.mean(precision_SVM2)
print(mean_precision_SVM2)
recall_SVM2 = cross_val_score(clf_SVM2, X_matrix2, y2, cv = k_fold, scoring = 'recall_macro')
mean_recall_SVM2 = np.mean(recall_SVM2)
print(mean_recall_SVM2)
f1Score_SVM2 = cross_val_score(clf_SVM2, X_matrix2, y2, cv = k_fold, scoring = 'f1_macro')
mean_f1Score_SVM2 = np.mean(f1Score_SVM2)
print("Mean f1Score with SVM - linear, C=1, on Test data: ", mean_f1Score_SVM2, "\n")