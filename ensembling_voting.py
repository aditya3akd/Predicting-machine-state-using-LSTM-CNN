import pandas as pd
import numpy as np
import statistics
# df=pd.DataFrame()
#df1=pd.read_csv('truelable_predlable_10k_data.csv')
# df2=pd.read_csv('truelable_predlable_10k_10mins.csv')
# df3=pd.read_csv('preds_autogluon_10000.csv')
# df4=pd.read_csv('last10000_predictions.csv')
# df5=pd.read_csv('conv1d_LSTM_10K_10M.csv')
# df['truelable_predlable_10k_data']=df1['pred']
# df['truelable_predlable_10k_10mins']=df2['pred']
# df['preds_autogluon_10000']=df3['State']
# df['last10000_predictions']=df4['predict']
# df['conv1d_LSTM_10K_10M']=df5['label']
# df.to_csv('ensemble.csv')
df=pd.read_csv('ensemble_withlabel_1000.csv')
x=[]
for i in range (0,1000):
    l=np.array(df.iloc[i:i+1,0:]).reshape(-1)
    res=statistics.mode(l)
    x.append(res)
df['finallabel']=x

df.to_csv('final_ensemblefor1000.csv')
# print(df['finallabel'])
from sklearn.metrics import accuracy_score

# df=pd.read_csv('final_ensemble.csv')
# df1=pd.read_csv('train_on_10thminute.csv')
# df['actual']=np.array(df1.iloc[40000:,25:26]).reshape(-1)
# print(df['actual'])
#df.to_csv('ensemble_withlabel.csv')
# print(accuracy_score(df['finallabel'],df['actual']))
