import pandas as pd
import numpy as np
import csv

df=pd.read_csv("D:\\bianyiqi\\学习满意度数据挖掘\\数据的预处理\\预处理后的问卷数据.csv")
data=np.array(df.values)
# print(data[0])
height,weight=data.shape

len=height


res=[]
for i in range(0,len):
    res1=[]
    tmp1=0
    tmp2=0
    tmp3=0
    for j in range(13,22):
        tmp1+=data[i][j]
    for j in range(22,31):
        tmp2+=data[i][j]
    for j in range(31,41):
        tmp3+=data[i][j]
    res1.append(round(tmp1 / 9, 4))
    res1.append(round(tmp2 / 9, 4))
    res1.append(round(tmp3 / 10, 4))
    res.append(res1)
# res=np.array(res)

print(res[0])



header = ['学习自主','学习能力','学习归属']

log_path = '相关性学习满意度.csv'
file = open(log_path, 'w',  newline='',encoding="utf-8")
csv_writer = csv.writer(file)
csv_writer.writerow(header)
for i in range(0,len):
    csv_writer.writerow(res[i])
file.close()





