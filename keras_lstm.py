from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import h5py

np.random.seed(7)

datafile = raw_input("Pandas dataframe to open: ")
df = pd.read_hdf("%s.h5" % datafile)
X = df[3:].as_matrix().T
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X = np.expand_dims(X, axis = 2)
Y = df.iloc[1:3].as_matrix().T
print np.shape(X),np.shape(Y)
# train_data = np.expand_dims(np.transpose(Y[:,:int(0.67 * len(Y[0]))]),axis = 2)
# test_data = np.expand_dims(np.transpose(Y[:,int(0.67 * len(Y[0])):])),axis = 2)
# train_var = np.expand_dims(np.transpose(test_total[:,:int(0.67 * len(Y[0]))])),axis = 2)
# test_var = np.expand_dims(np.transpose(test_total[:,int(0.67 * len(Y[0])):])),axis = 2)
# print np.shape(train_var), np.shape(test_var), np.shape(train_data), np.shape(test_data)
# train, test, index_train, index_test = train_test_split(np.transpose(dataset), test)
model = Sequential()
model.add(LSTM(10,input_dim = 1))
model.add(Dense(2,activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, Y, nb_epoch=3, batch_size=64)
scores = model.evaluate(X, Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
