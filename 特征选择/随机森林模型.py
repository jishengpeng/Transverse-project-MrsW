import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import csv

df=pd.read_csv("特征选择数据.csv",header=None)
df.columns = ['Class label', '学习动机', '自我效能感', '自我调节', '教师支持', '课程资源','平台支持', '同伴支持']
# print(df.head(5))
# # 标签类别
# set(df['Class label'])  #{1, 2, 3}
# print(df.shape) # (178, 14)
# # 统计缺失值
# print(df.isna().sum())

x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# print(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2022)
feat_labels = df.columns[1:]
# print(feat_labels)
forest = RandomForestClassifier(n_estimators=3000, random_state=0, max_depth=3)
forest.fit(x_train, y_train)
score = forest.score(x_test, y_test)# score=0.98148
print(score)
importances = forest.feature_importances_
print(importances)
indices = np.argsort(importances)[::-1] # 下标排序
for f in range(x_train.shape[1]):   # x_train.shape[1]=13
    print("%2d) %-*s %f" % \
          (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


#写进文件中
biaoti=['学习动机', '自我效能感', '自我调节', '教师支持', '课程资源','平台支持', '同伴支持']
log_path = '特征选择结果.csv'
file = open(log_path, 'w', newline='',encoding="utf-8")
csv_writer = csv.writer(file)
csv_writer.writerow(biaoti)
csv_writer.writerow(importances)
file.close()




