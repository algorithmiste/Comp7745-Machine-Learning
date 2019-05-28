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

cols = np.delete(np.asarray(titantic.columns),[2,5])
cols = cols.tolist()
titantic = MultiColumnLabelEncoder(columns = cols).fit_transform(titantic)
titantic_temp = titantic.drop(['survived', 'embarked'], axis = 1)
le = preprocessing.LabelEncoder()
embarked_np = le.fit_transform(np.asarray(titantic['embarked']).tolist())
embarked_df = pd.DataFrame(embarked_np, columns = ['embarked'])
titantic = pd.concat([titantic_temp, embarked_df, titantic['survived']], axis=1)

#titantic after pre-processing
titantic_X = titantic.drop(['survived'], axis = 1)
titantic_y = titantic['survived']

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
# clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 10)
# score = cross_val_score(clf, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'precision_macro')

#pruning factor = 10
clf10 = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 10)
precision10 = cross_val_score(clf10, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'precision_macro')
avgp10 = np.mean(precision10)
recall10 = cross_val_score(clf10, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'recall_macro')
avgr10 = np.mean(recall10)

#pruning factor = 20
clf20 = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 20)
precision20 = cross_val_score(clf20, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'precision_macro')
recall20 = cross_val_score(clf20, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'recall_macro')
avgp20 = np.mean(precision20)
avgr20 = np.mean(recall20)

#pruning factor = 30
clf30 = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 30)
precision30 = cross_val_score(clf30, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'precision_macro')
recall30 = cross_val_score(clf30, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'recall_macro')
avgp30 = np.mean(precision30)
avgr30 = np.mean(recall30)

#pruning factor = 40
clf40 = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 40)
precision40 = cross_val_score(clf40, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'precision_macro')
recall40 = cross_val_score(clf40, titantic_X, titantic_y, cv=k_fold, n_jobs=1, scoring= 'recall_macro')
avgp40 = np.mean(precision40)
avgr40 = np.mean(recall40)

x_coords = np.array([avgp10, avgp20, avgp30, avgp40])
y_coords = np.array([avgr10, avgr20, avgr30, avgr40])
types = ['10', '20', '30', '40']
l = "annotated with min_samples_leaf"

fig = plt.figure()
ax = plt.axes()
ax.margins()
ax.plot(x_coords, y_coords, 'bo', label = l)
ax.legend(loc = 'best')
props = {
    'title': 'Precision vs. Recall for DecisionTreeClassifier with 5-fold CV',
    'xlabel': 'Precision',
    'ylabel': 'Recall'
    }
ax.set(**props)

for i, txt in enumerate(types):
    ax.annotate(txt, (x_coords[i], y_coords[i]))
# for i,type in enumerate(types):
#     x = x_coords[i]
#     y = y_coords[i]
#     plt.scatter(x, y, marker='x', color='blue')
#     plt.text(x+0.3, y+0.3, type, fontsize=9)
plt.show()
#plot and add label to each data point

# precision_macro, recall_macro, f1_macro


