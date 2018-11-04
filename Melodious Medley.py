#Data Preprocessing

#Importing the Libraries
import numpy as np
import pandas as pd

############################################################################################################################################

#Importing the training dataset
dataset=pd.read_csv('train.csv')#File path

#Droping the "ts_listen" column from the dataset coz it's statistically insignificant variable
dataset.drop(["ts_listen"],axis=1,inplace=True)

#Loading dependent variable data
y_train=dataset.iloc[:,-2].values

#Droping the "is_listened" column from the dataset coz it's dependent variable
dataset.drop(["is_listened"],axis=1,inplace=True)

#Loading the features of training data
x_train=dataset.iloc[:,:].values

############################################################################################################################################

#Importing the testing dataset
testdata=pd.read_csv("test.csv") #File path

#Droping the "ts_listen" column from the dataset coz it's statistically insignificant variable
testdata.drop(["ts_listen"],axis=1,inplace=True)

#Loading the features of test data
x_test_original=testdata.iloc[:,:].values

############################################################################################################################################

#Feature Scaling
from sklearn.preprocessing import StandardScaler#importing the scaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test_original)

############################################################################################################################################

#Fitting Random Forest Classifier to the training set
from sklearn.ensemble import RandomForestClassifier#importing the classifier
classifier=RandomForestClassifier(n_estimators=30,criterion="entropy",min_samples_split=3,random_state=0)
classifier.fit(x_train,y_train)

############################################################################################################################################

#Predicting the test set results
y_pred=classifier.predict(x_test)

############################################################################################################################################

ID=x_test_original[:,-1]#copying the ID variable from the test dataset 
is_listened=y_pred#copying the predicted results
submission_array=np.array([ID,is_listened])#Concatinating the ID and predicted results array into a two dimensional array
submission_array=submission_array.T

'''Converting the array into a csv file(using Pandas)
It's too easy and fast
PS: This is why I love Pandas :) '''

df = pd.DataFrame(submission_array)
df.to_csv("Final_Submission.csv")

