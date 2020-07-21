import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Heart_Disease.csv')

y = dataset.iloc[:, -1].values
X = dataset.iloc[:, :-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 


from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import regularizers

classifier = Sequential()

classifier.add(Dense(input_dim = 13, kernel_regularizer=regularizers.l2(0.001), output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 128, kernel_regularizer=regularizers.l2(0.001),activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

epochs = 30

history = classifier.fit(X_train, y_train, epochs = epochs, batch_size = 25)


y_pred = classifier.predict(X_test)

for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0


from sklearn.metrics import roc_auc_score

ras = roc_auc_score(y_test, y_pred)

print(ras)

from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred)

print(cr)

