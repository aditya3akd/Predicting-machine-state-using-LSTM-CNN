import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
df=pd.read_csv('train_on_10thminute.csv')
X_train=df.iloc[0:40000,0:25]
y_train=df.iloc[0:40000,25:26]
X_test=df.iloc[40000:,0:25]
y_test=df.iloc[40000:,25:26]
import xgboost as xg
mf=xg.XGBClassifier()
mf.fit(X_train, y_train)
print("score on test: " + str(mf.score(X_test, y_test)))
print("score on train: "+ str(mf.score(X_train, y_train)))
y_pred=mf.predict(X_test)
cf=metrics.confusion_matrix(y_test,y_pred, labels=[0,1,2,3])
cf_dis=metrics.ConfusionMatrixDisplay(confusion_matrix=cf)
cf_dis.plot()
plt.savefig('Xgboost plot ')
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=25, max_depth=9)
rf.fit(X_train, y_train)
y_pred=mf.predict(X_test)
cf=metrics.confusion_matrix(y_test,y_pred, labels=[0,1,2,3])
cf_dis=metrics.ConfusionMatrixDisplay(confusion_matrix=cf)
cf_dis.plot()
plt.savefig('Random Forest Classifier')
plt.show()

print("score on test: " + str(rf.score(X_test, y_test)))
print("score on train: "+ str(rf.score(X_train, y_train)))
from sklearn.svm import LinearSVC
svm=LinearSVC(C=0.01)
svm.fit(X_train,np.array(y_train).reshape(40000,))
y_pred=mf.predict(X_test)
cf=metrics.confusion_matrix(y_test,y_pred, labels=[0,1,2,3])
cf_dis=metrics.ConfusionMatrixDisplay(confusion_matrix=cf)
cf_dis.plot()
plt.savefig('Support vector clasifier')
plt.show()

print("score on test: " + str(svm.score(X_test, np.array(y_test).reshape(10000,))))
