import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D,GlobalMaxPool1D, Dropout,SpatialDropout1D, SimpleRNN, LSTM, GRU,CuDNNGRU, CuDNNLSTM
from keras.layers import Flatten, Bidirectional
from sklearn.metrics import confusion_matrix
from keras.preprocessing import sequence
import keras

df=pd.read_csv('train_on_10thminute.csv')
X_train=df.iloc[0:40000,0:25]
y_train=df.iloc[0:40000,25:26]

X_test=df.iloc[40000:,0:25]
y_test=df.iloc[40000:,25:26]
df2=pd.read_csv('test_10th_minute.csv')
xtest=df2.iloc[:,:]
model3 = Sequential()
model3.add(Conv1D(256, 3, activation='PReLU', input_shape=(25, 1), padding='same'))
model3.add(MaxPool1D(1))
model3.add(Dropout(0.1))
model3.add(Conv1D(128, 2, activation='PReLU', padding='same'))
model3.add(MaxPool1D(2))
model3.add(Dropout(0.1))
model3.add((LSTM(256,return_sequences=True)))
model3.add((LSTM(64,return_sequences=True)))
model3.add(Flatten())
model3.add(Dense(32, activation='PReLU'))
model3.add(Dense(10, activation='PReLU'))
#     model.add(Dropout(0.1))
model3.add(Dense(4, activation='softmax'))
model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size = 128, epochs =20, verbose = 1)
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
from sklearn import metrics
y_pred=model3.predict(xtest)
y1_pred=model3.predict(X_test)
labels=[0,1,2,3]
lb.fit(labels)
pred=lb.inverse_transform((y_pred))
pred1=lb.inverse_transform((y1_pred))
df4=pd.DataFrame()
df4['predict']=pred
df4.to_csv('sai_predictions_1000.csv')
df5=pd.DataFrame()
df5['predict']=pred1

df5.to_csv('sai_predictions_ten_thousand.csv')