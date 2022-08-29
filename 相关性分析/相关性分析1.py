import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages



plt.rcParams['font.sans-serif']=['SimHei'] # 用黑体显示中文

df=pd.read_csv('相关性学习满意度.csv')
new_df=df.corr(method='pearson')
# new_df=df.corr(method='kendall')
# new_df=df.corr(method='spearman')
print(new_df)


# 引入seaborn库
plt.figure()
sns.heatmap(new_df,annot=True, vmax=1, square=True,cmap='Blues')#绘制new_df的矩阵热力图


pp = PdfPages('相关性学习满意度(皮尔森系数)' + '.pdf')
plt.savefig(pp, format='pdf', bbox_inches='tight')
pp.close()

plt.show()#显示图片
