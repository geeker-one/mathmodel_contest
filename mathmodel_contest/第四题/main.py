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

def fit_plane(x,y,Z):
    # 一个nx3的矩阵
    xyz = list()
    for i in range(len(x)):
        for j in range(len(y)):
            xyz.append((x[i],y[j],Z[j][i]))
    xyz = np.array(xyz)
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    N=len(x)

    A = np.array([[sum(x ** 2), sum(x * y), sum(x)],
                  [sum(x * y), sum(y ** 2), sum(y)],
                  [sum(x), sum(y), N]])

    B = np.array([[sum(x * z), sum(y * z), sum(z)]])

    # 求解
    X = np.linalg.solve(A, B.T)
    print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (X[0][0], X[1][0], X[2][0]))
    return X[0][0],X[1][0],X[2][0]

Para = fit_plane(x,y,Z)
normal_vector = np.array([-Para[0],-Para[1],1])
D0 = Para[0] * x[len(x)//2] + Para[1] * y[len(y)//2] + Para[2]
print(D0)

def calc_angle(v1,v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    dot_product = np.dot(v1,v2)
    cos_angle = dot_product / (norm1*norm2)
    # 注意是rad值
    angle = np.arccos(cos_angle)
    return angle

angle = calc_angle(normal_vector,np.array([0,0,1]))

# -------------------------结果展示-------------------------------
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.set_xlabel("x")
# ax1.set_ylabel("y")
# ax1.set_zlabel("z")
# # ax1.scatter(x, y, z, c='r', marker='o')
# x_p = 1852*np.arange(0,4.01,0.02)
# y_p = 1852*np.arange(0,5.01,0.02)
# x_p, y_p = np.meshgrid(x_p, y_p)
# z_p = X[0] * x_p + X[1] * y_p + X[2]
# ax1.plot_wireframe(x_p, y_p, z_p, rstride=10, cstride=10)
# plt.show()

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

