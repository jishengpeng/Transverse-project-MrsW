import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

df=pd.read_csv("D:\\bianyiqi\\学习满意度数据挖掘\\特征选择\\特征选择数据.csv",header=None)
df.columns = ['Class label', '学习动机', '自我效能感', '自我调节', '教师支持', '课程资源','平台支持', '同伴支持']
# print(df.head(5))
# # 标签类别
# set(df['Class label'])  #{1, 2, 3}
# print(df.shape) # (178, 14)
# # 统计缺失值
# print(df.isna().sum())

x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 加载数据
# iris = datasets.load_iris()

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 转换为Dataset数据格式
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

# 参数
params = {
    'learning_rate': 0.1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'max_depth': 4,
    'objective': 'regression',  # 目标函数
    # 'num_class': 16,
    'verbose':-1,
}

# print(train_data)

# 模型训练
gbm = lgb.train(params, train_data, valid_sets=[validation_data])

# 模型预测
y_pred = gbm.predict(X_test)
# y_pred = [list(x).index(max(x)) for x in y_pred]
# print(classification_report(y_test,y_pred))
print(y_pred)