import pandas as pd
import numpy as np
import csv

df=pd.read_csv("D:\\bianyiqi\\学习满意度数据挖掘\\数据的预处理\\预处理后的问卷数据.csv")
data=np.array(df.values)
# print(data[0])
height,weight=data.shape

len=height

header = ['总体满意度','LM1','LM2','LM3','LM4','LM5','LM6','LM7','LM8','LE1','LE2','LE3','LE4','LE5','LE6','LE7','LE8','LE9',
          'SE1','SE2','SE3','SE4','SE5','SE6','SR1','SR2','SR3','SR4','SR5','SR6','TR1','TR2','TR3','TR4','TR5',
          'TR6','TR7','CR1','CR2','CR3','CR4','PM1','PM2','PM3','PM4','PM5','PM6','PM7','PR1','PR2','PR3','PR4','PR5','PR6']

log_path = '预测数据.csv'
file = open(log_path, 'w',  newline='',encoding="utf-8")
csv_writer = csv.writer(file)
csv_writer.writerow(header)


for i in range(0,len):
    res=[]
    sum=0
    sum=data[i][41]+data[i][42]+data[i][43]
    res.append(sum)
    for j in range(44,97):
        res.append(data[i][j])
    csv_writer.writerow(res)


file.close()





