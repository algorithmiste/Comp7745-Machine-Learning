#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


titantic = pd.read_csv("C:/Users/casey/AnacondaProjects/data/Comp 7745/titanic.csv")
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


df_embarked = pd.get_dummies(titantic['embarked'])
print(titantic)
#titantic after pre-processing
cols = np.delete(np.asarray(titantic.columns),[2,5])
cols = cols.tolist()
print(cols)
# print(cols)
titantic = MultiColumnLabelEncoder(columns = cols).fit_transform(titantic)
# print(titantic)

titantic_temp = titantic.drop(['survived', 'embarked'], axis = 1)
le = preprocessing.LabelEncoder()
embarked_np = le.fit_transform(np.asarray(titantic['embarked']).tolist())
embarked_df = pd.DataFrame(embarked_np, columns = ['embarked'])

titantic = pd.concat([titantic_temp, embarked_df, titantic['survived']], axis=1)

titantic_X = titantic.drop(['survived'], axis = 1)
titantic_y = titantic['survived']

X_train, X_test, y_train, y_test = train_test_split(titantic_X, titantic_y, test_size=0.25, random_state=33)

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

print(np.dtype(y_train))

clf = tree.DecisionTreeClassifier(criterion = ['gini', 'entropy'])
# model = clf.fit(X_train, y_train)

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)

# predict probabilities
probs = clf.predict_proba(X_test) #.predict_proba(testX)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(testX)
