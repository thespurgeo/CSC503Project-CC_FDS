
# Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt
import config
import ovs
import uns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Manager


import warnings
warnings.filterwarnings("ignore")

def report_metric(params,label, pred):
      print("Average_precision_score of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
            metrics.average_precision_score(label, pred))



# Loading the Data
col_names=config.col_names2
feature_cols=config.feature_cols2
path_to_dataset=config.path_to_dataset2
target=config.target2

prob_data = pd.read_csv(path_to_dataset,skiprows=[0], header=None, names=col_names)
prob_data.head()


#split dataset in features and target variable
X = prob_data[feature_cols] # Features
y = prob_data[target] # Target variable

print('\nImbalanced dataset Results: \n')

#preprocessing for non-standardized data
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=7,shuffle=True,stratify=y) # 80% training and 20% test
print('Results with neither Oversampling nor Undersampling the non-standardized data set')
clfs = MLPClassifier(hidden_layer_sizes=(15,), random_state=7, max_iter=1000,tol=1e-6,learning_rate='adaptive').fit(Xr_train, yr_train)
ytr_pred = clfs.predict(Xr_train)
params = clfs.get_params()
print("Train accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
     metrics.balanced_accuracy_score(yr_train, ytr_pred))
yr_pred = clfs.predict(Xr_test)
params = clfs.get_params()
print("Test accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
      metrics.balanced_accuracy_score(yr_test, yr_pred))

#feature scaling, Fit only to the training data
scaler = StandardScaler()

#Oversampling=False, normalization=False, standardization=False
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=7,shuffle=True,stratify=y) # 80% training and 20% test
print('\nResults with neither Oversampling nor Undersampling the standardized data set')
scaler.fit(Xr_train)
Xr_train = scaler.transform(Xr_train)
Xr_test = scaler.transform(Xr_test)
clfs = MLPClassifier(hidden_layer_sizes=(15,), random_state=7, max_iter=1000,tol=1e-6,learning_rate='adaptive').fit(Xr_train, yr_train)
ytr_pred = clfs.predict(Xr_train)
params = clfs.get_params()
print("Train accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
      metrics.balanced_accuracy_score(yr_train, ytr_pred))

yr_pred = clfs.predict(Xr_test)
params = clfs.get_params()
print("Test accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
      metrics.balanced_accuracy_score(yr_test, yr_pred))


print('\nBalanced dataset Results: \n')

#Oversampling=True, normalization=False, standardization=False
Xo_train, Xo_test, yo_train, yo_test = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=7,shuffle=True,stratify=y) # 80% training and 20% test
scaler.fit(Xo_train)
Xo_train = scaler.transform(Xo_train)
Xo_test = scaler.transform(Xo_test)
Xo_train,yo_train=ovs.ovsample(Xo_train,yo_train)
print('Results with Oversampling the standardized data set without normalization ')
clfs = MLPClassifier(hidden_layer_sizes=(15,), random_state=7, max_iter=1000,tol=1e-6,learning_rate='adaptive').fit(Xo_train, yo_train)
yor_pred = clfs.predict(Xo_train)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
#print("Train accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
 #     metrics.classification_report(yo_train, yor_pred))
print("Train ")
report_metric(params,yo_train, yor_pred)

yo_pred = clfs.predict(Xo_test)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
#print("Test accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
#      metrics.classification_report(yo_test, yo_pred))
print("Test ")
report_metric(params,yo_test, yo_pred)

#Undersampling=True, normalization=False, standardization=False
Xo_train, Xo_test, yo_train, yo_test = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=7,shuffle=True,stratify=y) # 80% training and 20% test
scaler.fit(Xo_train)
Xo_train = scaler.transform(Xo_train)
Xo_test = scaler.transform(Xo_test)
Xo_train,yo_train=uns.undersample(Xo_train,yo_train)
print('\nResults with Undersampling the standardized data set without normalization ')
clfs = MLPClassifier(hidden_layer_sizes=(15,), random_state=7, max_iter=1000,tol=1e-6,learning_rate='adaptive').fit(Xo_train, yo_train)
yor_pred = clfs.predict(Xo_train)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
#print("Train accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
#      metrics.classification_report(yo_train, yor_pred))
print("Train ")
report_metric(params,yo_train, yor_pred)

yo_pred = clfs.predict(Xo_test)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
#print("Test accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
#      metrics.classification_report(yo_test, yo_pred))
print("Test ")
report_metric(params,yo_test, yo_pred)



#Oversampling=True, normalization=True, standardization=False
Xn_train, Xn_test, yn_train, yn_test = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=7,shuffle=True,stratify=y) # 80% training and 20% test
scaler.fit(Xn_train)
Xn_train = scaler.transform(Xn_train)
Xn_test = scaler.transform(Xn_test)
Xn_train = preprocessing.normalize(Xn_train)
Xn_test = preprocessing.normalize(Xn_test)
Xn_train,yn_train=ovs.ovsample(Xn_train,yn_train)
print('\nResults with Normalizing the standardized data set (after Oversampling)')
clfs = MLPClassifier(hidden_layer_sizes=(15,), random_state=7, max_iter=1000,tol=1e-6,learning_rate='adaptive').fit(Xn_train, yn_train)
ynr_pred = clfs.predict(Xn_train)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
#print("Train accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
#      metrics.classification_report(yn_train, ynr_pred))
print("Train ")
report_metric(params,yn_train, ynr_pred)

yn_pred = clfs.predict(Xn_test)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
#print("Test accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
#      metrics.classification_report(yn_test, yn_pred))
print("Test ")
report_metric(params,yn_test, yn_pred)


#Undersampling=True, normalization=True, standardization=False
Xn_train, Xn_test, yn_train, yn_test = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=7,shuffle=True,stratify=y) # 80% training and 20% test
scaler.fit(Xn_train)
Xn_train = scaler.transform(Xn_train)
Xn_test = scaler.transform(Xn_test)
Xn_train = preprocessing.normalize(Xn_train)
Xn_test = preprocessing.normalize(Xn_test)
Xn_train,yn_train=uns.undersample(Xn_train,yn_train)
print('\nResults with Normalizing the standardized data set (after Undersampling)')
clfs = MLPClassifier(hidden_layer_sizes=(15,), random_state=7, max_iter=1000,tol=1e-6,learning_rate='adaptive').fit(Xn_train, yn_train)
ynr_pred = clfs.predict(Xn_train)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
#print("Train accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
#      metrics.classification_report(yn_train, ynr_pred))
print("Train ")
report_metric(params,yn_train, ynr_pred)

yn_pred = clfs.predict(Xn_test)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
#print("Test accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
#      metrics.classification_report(yn_test, yn_pred))
print("Test ")
report_metric(params,yn_test, yn_pred)


'''
#Oversampling=True, normalization=False, standardization=True
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=7,shuffle=True,stratify=y) # 80% training and 20% test
scaler.fit(Xs_train)
Xs_train = scaler.transform(Xs_train)
Xs_test = scaler.transform(Xs_test)
Xs_train = preprocessing.scale(Xs_train)
Xs_test = preprocessing.scale(Xs_test)
Xs_train,ys_train=ovs.ovsample(Xs_train,ys_train)
print('Results with Standardizing the data set (after Oversampling)\n')
clfs = MLPClassifier(hidden_layer_sizes=(15,), random_state=7, max_iter=1000,tol=1e-6,learning_rate='adaptive').fit(Xs_train, ys_train)
ysr_pred = clfs.predict(Xs_train)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
print("Train accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
      metrics.classification_report(ys_train, ysr_pred))
ys_pred = clfs.predict(Xs_test)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
print("Test accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
      metrics.classification_report(ys_test, ys_pred))

#Undersampling=True, normalization=False, standardization=True
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X, y, test_size=0.2, train_size=0.8,random_state=7,shuffle=True,stratify=y) # 80% training and 20% test
scaler.fit(Xs_train)
Xs_train = scaler.transform(Xs_train)
Xs_test = scaler.transform(Xs_test)
Xs_train = preprocessing.scale(Xs_train)
Xs_test = preprocessing.scale(Xs_test)
Xs_train,ys_train=uns.undersample(Xs_train,ys_train)
print('Results with Standardizing the data set (after Undersampling)\n')
clfs = MLPClassifier(hidden_layer_sizes=(15,), random_state=7, max_iter=1000,tol=1e-6,learning_rate='adaptive').fit(Xs_train, ys_train)
ysr_pred = clfs.predict(Xs_train)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
print("Train accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
      metrics.classification_report(ys_train, ysr_pred))
ys_pred = clfs.predict(Xs_test)
params = clfs.get_params()
# print("The accuracy of neural network with hidden layer sizes of ",params['hidden_layer_sizes']," is ",metrics.accuracy_score(y_test, y_pred))
print("Test accuracy of neural network with hidden layer sizes of ", params['hidden_layer_sizes'], " is \n",
      metrics.classification_report(ys_test, ys_pred))
'''