import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
df=pd.read_csv('ensemble_withlabel_10000.csv')
X_train=df.iloc[0:8000,0:6]
y_train=df.iloc[0:8000,6:7]
X_test=df.iloc[8000:,0:6]
y_test=df.iloc[8000:,6:7]
df1=pd.read_csv('ensemble_withlabel_1000.csv')
xtest=df1.iloc[:,0:6]
model=Sequential()
model.add(Dense(12, input_shape=(6,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit((X_train), (y_train), epochs=1,batch_size=64)
_, accuracy =model.evaluate((X_test),(y_test))
# Accuracy was 79.31
# df=pd.DataFrame()
#y_pred=model.predict(xtest)
# from sklearn import preprocessing
# lb = preprocessing.LabelBinarizer()
# from sklearn import metrics
# y_pred=model.predict(xtest)
# labels=[0,1,2,3]
# lb.fit(labels)
# pred=lb.inverse_transform((y_pred))
# print(pred)
# df3=pd.DataFrame()
# df3['predict']=pred
# df3.to_csv('last1000_nn.csv')
# from sklearn.metrics import accuracy_score
# df1=pd.read_csv('last1000_nn.csv')
df1=pd.read_csv('final_predictions.csv')
df2=pd.read_csv('ensemble_withlabel_1000.csv')
print(accuracy_score(df1['finallabel'],df2['predict1']))

# df=pd.read_csv('final_predictions.csv')
# # for i in range (50000,51000):
# #     f.append(i)
# # df['Sample ID']=f
# df.to_csv('finalsubmission.txt',index=False,header=False)





