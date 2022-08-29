import pandas as pd
import numpy as np
import csv

df=pd.read_csv("预处理后的问卷数据.csv");
data=np.array(df.values)
# print(data[0])
height,weight=data.shape


#一边进行预处理，一边写入文件中
len=1

for i in range(0,len):
    print(data[i])

