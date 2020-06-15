import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense

labels = pd.read_csv('C:\GitHub\DL_Python\Breast_Cancer\labels.csv')
inputs = pd.read_csv('C:\GitHub\DL_Python\Breast_Cancer\inputs.csv')

inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.25)

model = Sequential([
    Dense(units = 16, activation = 'relu', 
          kernel_initializer = 'random_uniform', 
          input_dim = 30),

    Dense(units = 1, activation = 'sigmoid')])

inputs_test.shape

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

model.fit(inputs_train, labels_train,
          batch_size = 10,
          epochs = 100)

preds = np.where(model.predict(inputs_test) > 0.5, 1, 0)
precisao = accuracy_score(labels_test, preds)
cm = confusion_matrix(labels_test, preds)

model.evaluate(inputs_test, labels_test)