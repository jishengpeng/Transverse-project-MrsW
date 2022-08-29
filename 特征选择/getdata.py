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
    tmp1 = 0
    tmp2 = 0
    tmp3 = 0
    tmp4 = 0
    tmp5 = 0
    tmp6 = 0
    tmp7 = 0
    tmp8 = 0
    tmp9 = 0
    for j in range(44,52):
        tmp1+=data[i][j]
    for j in range(52,61):
        tmp2+=data[i][j]
    for j in range(61,67):
        tmp3+=data[i][j]
    for j in range(67,73):
        tmp4+=data[i][j]
    for j in range(73,80):
        tmp5+=data[i][j]
    for j in range(80,84):
        tmp6+=data[i][j]
    for j in range(84,91):
        tmp7+=data[i][j]
    for j in range(91,97):
        tmp8+=data[i][j]
    for j in range(41,44):
        tmp9+=data[i][j]
    res1.append(tmp9)
    res1.append(round(tmp1 / 8, 4))
    # res1.append(round(tmp2 / 9, 4))
    res1.append(round(tmp3 / 6, 4))
    res1.append(round(tmp4 / 6, 4))
    res1.append(round(tmp5 / 7, 4))
    res1.append(round(tmp6 / 4, 4))
    res1.append(round(tmp7 / 7, 4))
    res1.append(round(tmp8 / 6, 4))
    res.append(res1)
# res=np.array(res)

print(res[0])


log_path = '特征选择数据.csv'
file = open(log_path, 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(file)


for i in range(0,len):
    csv_writer.writerow(res[i])
file.close()






