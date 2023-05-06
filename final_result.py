import pandas as pd
import numpy as np
df1=pd.read_csv('test_10th_minute.csv')

import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D,GlobalMaxPool1D, Dropout,SpatialDropout1D, SimpleRNN, LSTM, GRU,CuDNNGRU, CuDNNLSTM
from keras.layers import Flatten, Bidirectional
from sklearn.metrics import confusion_matrix
from keras.preprocessing import sequence
import keras
import pandas as pd
# df=pd.read_csv('train_on_10thminute.csv')
# X_train=df.iloc[0:40000,0:25]
# y_train=df.iloc[0:40000,25:26]
# #print(type(df))
# X_test=df1.iloc[0:,0:25]
# #y_test=df.iloc[40000:,25:26]
# model = Sequential()
# model.add(Conv1D(256, 3, activation='relu', input_shape=(25, 1), padding='same'))
# model.add(MaxPool1D(2))
# model.add(SpatialDropout1D(0.2))
# model.add(Conv1D(32, 3, activation='relu', padding='same'))
# model.add(MaxPool1D(2))
# # model.add(SpatialDropout1D(0.1))
# model.add((Bidirectional(GRU(64,return_sequences=True))))
# model.add((Bidirectional(GRU(32,return_sequences=True))))
#
# model.add((Dense(32)))
# model.add(Flatten())
# model.add(Dense(64, activation='selu'))
# model.add(Dropout(0.08))
# model.add(Dense(4, activation='softmax'))
# #model.add(Dense(1))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit((X_train), y_train, batch_size=128, epochs=20, verbose=0)
# # _, accuracy = model.evaluate((X_test), (y_test))
# # print('Accuracy: %.2f' % (accuracy * 100))
# from sklearn import preprocessing
# lb = preprocessing.LabelBinarizer()
# from sklearn import metrics
# y_pred=model.predict(X_test)
# labels=[0,1,2,3]
# lb.fit(labels)
# pred=lb.inverse_transform((y_pred))
# print(pred)
# df3=pd.DataFrame()
# df3['predict']=pred
# df3.to_csv('prediction.csv')
# cf=metrics.confusion_matrix(y_test,pred, labels=[0,1,2,3])
# cf_dis=metrics.ConfusionMatrixDisplay(confusion_matrix=cf)
# cf_dis.plot()
# plt.savefig('lstm+cnn(1D)for 10th minute data')
# plt.show()
df2=pd.read_csv('prediction.csv')
df3=pd.read_csv('preds.csv')
c=0
# print(type(df3.iloc[2:3,0:1].values))
for i in range(0,1000):
    if df2.iloc[i:i+1,1:2].values==np.array(df3.iloc[i:i+1,0:1]).astype('int64'):
        c=c+1
print(c)
