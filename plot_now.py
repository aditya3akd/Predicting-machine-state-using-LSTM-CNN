import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
df1=pd.read_csv('normal_train.csv')
df2=pd.read_csv('train_on_10thminute.csv')
df1.dropna(inplace=True)
df3=pd.read_csv('mean_data.csv')
c=0

for j in range(0,25):
    a = []
    b = []
    c=0
    l=np.array(df2.iloc[0:50000,j:j+1]).reshape(-1)


    for k in l:

        if (math.floor(k * 10) + 1) / 10>1:
            a.append((math.floor(k * 10)) / 10 )
        else:
            a.append((math.floor(k * 10) + 1) / 10)


        b.append((np.array(df2.iloc[c:c + 1, 25:26]).reshape(-1))[0])
        print(len(b))
        c=c+1
    #l = [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1]
    #g = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
    df = pd.DataFrame()
    df['x'] = pd.Series(a)
    df['y'] = pd.Series(b)
    (df.groupby('x')['y'].value_counts(normalize=True)
     .unstack('y').plot.bar(stacked=True)
     )
    plt.show()






