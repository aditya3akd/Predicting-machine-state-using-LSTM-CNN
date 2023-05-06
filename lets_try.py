import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
import numpy as np
df=pd.read_csv('train.txt')
daf=pd.DataFrame(columns = ["SampleID","TimeStamp"]+list(range(1,26)))
daf.loc[0]=df.columns
df.columns=daf.columns
df=pd.concat([daf,df],ignore_index=True)


#df.dropna(inplace=True)


df2=pd.Series(df.iloc[:,2:27].iloc[0:10,:].values.reshape(1,-1)[0])



#first 20000 train, next 10000 test, accuracy 77%
#10000-30000 train,first 10000 test 76.38 accuracy
df4=pd.read_csv('final_train_250.csv')
print(df4)
df4.dropna(inplace=True)
df_final=pd.read_csv('final_train_250.csv')
X_train=np.array(df4.iloc[0:40000,0:250]).reshape(40000,1,10,25)
y_train=df4.iloc[0:40000,250:251]
X_val=np.array(df4.iloc[35000:40000,0:250]).reshape(5000,10,25)
y_val=df4.iloc[35000:40000,250:251]
print(type(df))
X_test=np.array(df4.iloc[40000:,0:250]).reshape(10000,1,10,25)
y_test=df4.iloc[40000:,250:251]
#model = Sequential()
#model.add(Dense(30, input_shape=(250,), activation='relu'))
#model.add(Dense(40, activation='relu'))
#model.add(Dense(40, activation='relu'))
#model.add(Dense(30, activation='relu'))
#model.add(Dense(4, activation='softmax'))
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train,y_train, epochs=100,batch_size=100)

print(type(df))
#_,  accuracy = model.evaluate(X_test,y_test)
#print('Accuracy: %.2f' % (accuracy*100))
#print(df_final)
# classi=Sequential()
# classi.add(SimpleRNN(units = 50, activation='relu', return_sequences=True, input_shape= (25,1)))
# classi.add(Dropout(0.2))
# classi.add(SimpleRNN(units = 60, activation='relu', return_sequences=True))
# classi.add(Dropout(0.2))
# classi.add(SimpleRNN(units = 70, activation='relu', return_sequences=True))
# classi.add(Dropout(0.2))
# classi.add(SimpleRNN(units = 50))
# classi.add(Dropout(0.2))
# classi.add(Dense(units = 4,activation='softmax'))
# classi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# model = KerasClassifier(model=classi,optimizer=tf.keras.optimizers.Adam(), verbose=0)
# # define the grid search parameters
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [5,10,20]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X_train, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
# classi.fit((X_train), (y_train), epochs=25,batch_size=128)
# _, accuracy = classi.evaluate((X_test),(y_test))
# # Accuracy was 79.31

#print('Accuracy: %.2f' % (accuracy*100))
# from sklearn.svm import LinearSVC
# svm=LinearSVC(C=0.0001)
# svm.fit(X,Y)
# print("score on test: " + str(svm.score(X1, Y1)))
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(max_depth=100)
# clf.fit(X_train, y_train)
# print("score on test: "  + str(clf.score(X_test, y_test)))
# print("score on train: " + str(clf.score(X_train, y_train)))
# from sklearn.ensemble import RandomForestClassifier
# # n_estimators = number of decision trees
# rf = RandomForestClassifier(n_estimators=100, max_depth=9)
# rf.fit(X_train, y_train)
# print("score on test: " + str(rf.score(X_test, y_test)))
# print("score on train: "+ str(rf.score(X_train, y_train)))
# import xgboost as xg
# mf=xg.XGBClassifier()
# mf.fit(X_train, y_train)
# print("score on test: " + str(mf.score(X_test, y_test)))
# print("score on train: "+ str(mf.score(X_train, y_train)))
# from keras.layers import LSTM
# regressor = Sequential()
#
# # Adding the input layerand the LSTM layer
# regressor.add(LSTM(units = 8, activation = 'relu', input_shape = (250, 1)))
#
#
# # Adding the output layer
# regressor.add(Dense(units = 4,activation='softmax'))
#
# # Compiling the RNN
# regressor.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Fitting the RNN to the Training set
# regressor.fit(X_train, y_train, batch_size = 128, epochs = 5, verbose = 0)
# _, accuracy = regressor.evaluate((X_test),(y_test))
# # Accuracy was 79.31
#
# print('Accuracy: %.2f' % (accuracy*100))
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
# model_cnn_lstm = Sequential()
# model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,25,1)))
# model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model_cnn_lstm.add(TimeDistributed(Flatten()))
# model_cnn_lstm.add(LSTM(50, activation='relu'))
# model_cnn_lstm.add(Dense(4,activation='softmax'))
# model_cnn_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# model_cnn_lstm.fit(X_train, y_train, batch_size = 128, epochs = 5, verbose = 0)
# _, accuracy = model_cnn_lstm.evaluate((X_test),(y_test))
# print('Accuracy: %.2f' % (accuracy*100))
from keras.preprocessing import sequence
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Conv1D,Conv2D, MaxPool1D,MaxPool2D ,Dropout, SimpleRNN, LSTM, GRU, MaxPooling2D
from keras.layers import Flatten, Reshape
from sklearn.metrics import confusion_matrix
from keras.preprocessing import sequence
model = Sequential()
model.add(Conv2D(30,5,5, activation='relu', input_shape=(1,10,25)))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (4,2), activation='tanh', padding='same'))
model.add(MaxPooling2D(3,3))

# model.add(Dropout(0.04))
# for layers in model.layers:
#     print(layers.output_shape)
# model.add(Reshape((64,4), input_shape=(None,1,4,64)))
# model.add(LSTM(64))
# model.add(Reshape((32,2), input_shape=(None,64)))
# model.add(LSTM(32))

model.add((Dense(32)))
model.add(Flatten())
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(4, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
model.fit((X_train), y_train, batch_size=32, epochs=2, verbose=0)
_, accuracy = model.evaluate((X_test), (y_test))
print('Accuracy: %.2f' % (accuracy * 100))
#print("Epochs:" + str(j))
#print("Dropout factor"+ str(i))


#y_pred=model.predict(X_test)


