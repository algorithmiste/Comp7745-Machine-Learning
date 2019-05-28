#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


wine = pd.read_csv("C:/Users/casey/AnacondaProjects/data/Comp 7745/wines.csv")
# print(wine)

wine_X = wine.drop('quality', axis = 1)
wine_y = wine['quality']
#validation fract: validation_fraction

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
# 1 layer
mlp1a = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, learning_rate_init = 0.01, validation_fraction = 0.1) # (100,), (100, 100,)
score1a = cross_val_score(mlp1a, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s1a = np.mean(score1a)
print(mean_s1a)

mlp1b = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, learning_rate_init = 0.01, validation_fraction = 0.3) # (100,), (100, 100,)
score1b = cross_val_score(mlp1b, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s1b = np.mean(score1b)
print(mean_s1b)
# precision_macro, recall_macro, f1_macro

mlp2a = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, learning_rate_init = 0.1, validation_fraction = 0.1) # (100,), (100, 100,)
score2a = cross_val_score(mlp2a, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s2a = np.mean(score2a)

mlp2b = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, learning_rate_init = 0.1, validation_fraction = 0.3 ) # (100,), (100, 100,)
score2b = cross_val_score(mlp2b, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s2b = np.mean(score2b)


mlp3a = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, learning_rate_init = 0.2, validation_fraction = 0.1) # (100,), (100, 100,)
score3a = cross_val_score(mlp3a, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s3a = np.mean(score3a)

mlp3b = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, learning_rate_init = 0.2, validation_fraction = 0.3) # (100,), (100, 100,)
score3b = cross_val_score(mlp3b, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s3b = np.mean(score3b)
# print(score)

# 2 layers
mlp4a = MLPClassifier(hidden_layer_sizes=(100,100,), max_iter=100, learning_rate_init = 0.01, validation_fraction = 0.1) # (100,), (100, 100,)
score4a = cross_val_score(mlp4a, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s4a = np.mean(score4a)

mlp4b = MLPClassifier(hidden_layer_sizes=(100,100,), max_iter=100, learning_rate_init = 0.01, validation_fraction = 0.3) # (100,), (100, 100,)
score4b = cross_val_score(mlp4b, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s4b = np.mean(score4b)

mlp5a = MLPClassifier(hidden_layer_sizes=(100, 100,), max_iter=100, learning_rate_init = 0.1, validation_fraction = 0.1) # (100,), (100, 100,)
score5a = cross_val_score(mlp5a, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s5a = np.mean(score5a)

mlp5b = MLPClassifier(hidden_layer_sizes=(100, 100,), max_iter=100, learning_rate_init = 0.1, validation_fraction = 0.3) # (100,), (100, 100,)
score5b = cross_val_score(mlp5b, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s5b = np.mean(score5b)


mlp6a = MLPClassifier(hidden_layer_sizes=(100, 100,), max_iter=100, learning_rate_init = 0.2, validation_fraction = 0.1) # (100,), (100, 100,)
score6a = cross_val_score(mlp6a, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s6a = np.mean(score6a)

mlp6b = MLPClassifier(hidden_layer_sizes=(100, 100,), max_iter=100, learning_rate_init = 0.2, validation_fraction = 0.3) # (100,), (100, 100,)
score6b = cross_val_score(mlp6b, wine_X, wine_y, cv = k_fold, n_jobs=1, scoring= 'f1_macro')
mean_s6b = np.mean(score6b)


mlpaArr = np.array([mean_s1a, mean_s2a, mean_s3a, mean_s4a, mean_s5a, mean_s6a])
mlpbArr = np.array([mean_s1b, mean_s2b, mean_s3b, mean_s4b, mean_s5b, mean_s6b])
print(mlpaArr)
print(mlpbArr)
# predictions = mlp.predict(X_test)
# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))