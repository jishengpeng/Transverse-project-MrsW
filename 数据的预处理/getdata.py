import pandas as pd
import numpy as np
import csv

df=pd.read_excel("../data.xlsx");
data=np.array(df.values)
# print(data[0])
height,weight=data.shape

log_path = '预处理后的问卷数据.csv'
file = open(log_path, 'w',  newline='',encoding="utf-8")
csv_writer = csv.writer(file)

#一边进行预处理，一边写入文件中
len=height

for i in range(0,len):
    print(data[i])
    data[i][15] = 6 - data[i][15]
    data[i][16] = 6 - data[i][16]
    data[i][17] = 6 - data[i][17]
    data[i][22] = 6 - data[i][22]
    data[i][23] = 6 - data[i][23]
    data[i][27] = 6 - data[i][27]
    data[i][33] = 6 - data[i][33]
    data[i][36] = 6 - data[i][36]
    data[i][37] = 6 - data[i][37]
    data[i][44] = 6 - data[i][44]
    data[i][54] = 6 - data[i][54]
    data[i][60] = 6 - data[i][60]
    data[i][75] = 6 - data[i][75]
    csv_writer.writerow(data[i])
file.close()
