import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense

labels = pd.read_csv('C:\GitHub\DL_Python\Breast_Cancer\labels.csv')
inputs = pd.read_csv('C:\GitHub\DL_Python\Breast_Cancer\inputs.csv')

inputs_test, inputs_train, lablesl_test, labels_train = train_test_split(inputs, labels, test_size=0.25)

model = Sequential([
    Dense(units = 16, activation = 'relu', 
          kernel_initializer = 'random_uniform', 
          input_dim = 30),

    Dense(units = 1, activation = 'sigmoid')])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
model.fit(inputs_train, labels_train,
          batch_size = 10,
          epochs = 100)