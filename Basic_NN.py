

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#import keras
import keras
from keras.models import Sequential 
from keras.layers import Dense

#intilizing ANN
classifier= Sequential()

#adding input layer and first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

#adding second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#adding output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#compling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train, y_train,batch_size=10, nb_epoch=100)

#predicting the test set
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#for specific data values
new_prediction= classifier.predict(sc.transform(np.array([[0.0 ,0 ,600 ,0,40 ,3 ,60000 ,2 ,1 ,1 ,50000]])))
new_prediction=(new_prediction>0.5)

#confusion matri
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)