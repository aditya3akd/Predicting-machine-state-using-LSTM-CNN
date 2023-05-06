import pandas as pd
df=pd.read_csv('final_train_250.csv')
print(df)
df2=df.iloc[0:50000,176:201]
print(df2)
df3=pd.read_csv('train_on_10thminute.csv')
df2['label']=df3['label']
df2.to_csv('train_on_8thminute.csv')