import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import leastsq


# 设置中文字体
plt.rcParams["font.sans-serif"]=["Source Han Serif CN"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

df = pd.read_excel('./附件.xlsx',header = 1,index_col=1)
df.drop('Unnamed: 0',axis = 1,inplace = True)
Z = np.array(df.values)
x = 1852*np.arange(0,4.01,0.02)
y = 1852*np.arange(0,5.01,0.02)
# x = np.arange(0,4.01,0.02)
# y = np.arange(0,5.01,0.02)
X,Y = np.meshgrid(x,y)

# 一个nx3的矩阵
xyz = list()
for i in range(len(x)):
    for j in range(len(y)):
        xyz.append((x[i],y[j],Z[j][i]))
xyz = np.array(xyz)


# 计算起伏特别小的那一条边最小二乘斜率
def fit_line():
    x_list = x
    y_list = Z[-1]
    def error(p,x,y):
        return y-line_func(p,x)
    def line_func(p,x):
        k,b= p
        return k*x+b
    result = leastsq(error,[0.01,1],args = (x_list,y_list))
    # 打印k
    # 发现k超级小
    return result[0]
def get_aver_line():
    return np.average(Z[-1])

