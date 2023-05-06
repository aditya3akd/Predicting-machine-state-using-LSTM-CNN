import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
df=pd.read_csv('normal_train.csv')
dx=pd.DataFrame()
import statsmodels.api as sm
# for i in range (0,550000,11):
#     l=[]
#     for j in range(3,28):
#         model=ARIMA(np.array(df.iloc[i:i+10,j:j+1]).reshape(-1), order=(1,0,2))
#         model_fit = model.fit()
#         a = model_fit.predict(start=0, end=10, dynamic=False)
#         l.append(a[10])

   # dx=dx.append(pd.Series(l),ignore_index=True)
#dx.to_csv('11thminute.csv')
x_train=[0,1,2,3,4,5,6,7,8,9]
for i in [0,1,2,3,4,5,6,7,8]:
    for j in [0,1,2,3,4,5,6,7,8]:
        for k in [0,1,2,3,4,5,6,7,8]:
            model = ARIMA(np.array(df.iloc[0:10, 3:4]).reshape(-1), order=(i,j,k))
            model_fit = model.fit()
            a = model_fit.predict(start=0, end=9, dynamic=False)
            # l.append(a[2])
            # l.append(a[4])
            # print(l)

            # print(a[9])
            #print("MAPE error:-")
            z = np.mean(np.abs(100 * a - 100 * np.array(df.iloc[0:10, 3:4]).reshape(-1)) / (
                        100 * np.abs(np.array(df.iloc[0:10, 3:4]).reshape(-1))))
            if z<2:
                print(z)
                print(i)
                print(j)
                print(k)




model = ARIMA(np.array(df.iloc[0:10,3:4]).reshape(-1), order=(1,1,1))
model_fit = model.fit()
a = model_fit.predict(start=0, end=9, dynamic=False)
#l.append(a[2])
#l.append(a[4])
#print(l)


            #print(a[9])
print("MAPE error:-")
z=np.mean(np.abs(100*a - 100*np.array(df.iloc[ 0:10,3:4]).reshape(-1)) /(100*np.abs(np.array(df.iloc[0:10,3:4]).reshape(-1))))
print(z)
# model=sm.tsa.statespace.SARIMAX(np.array(df.iloc[3:4, 0:10]).reshape(-1),order=(2, 0, 1),seasonal_order=(0,1,1,2))
# results=model.fit()
#
dp=pd.DataFrame()


#a=model_fit.predict(start=0,end=9,dynamic=False)
dp['sensor1']=np.array(df.iloc[0:10,3:4]).reshape(-1)
dp['ax']=a
dp[['sensor1', 'ax']].plot(figsize=(12, 8))
plt.show()
# for i in range(0,11):
#     print(a[i])