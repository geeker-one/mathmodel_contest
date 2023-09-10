import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from scipy import interpolate
from third import waves_group


# 设置中文字体
plt.rcParams["font.sans-serif"]=["Source Han Serif CN"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

def draw_fit_plane():
    df = pd.read_excel('./附件.xlsx',header = 1,index_col=1)
    df.drop('Unnamed: 0',axis = 1,inplace = True)
    Z = np.array(df.values)
    x = 1852*np.arange(0,4.01,0.02)
    y = 1852*np.arange(0,5.01,0.02)
    X,Y = np.meshgrid(x,y)



    x1= 1852*np.linspace(0,4,4000)
    y1= 1852*np.linspace(0,5,5000)
    X1,Y1 = np.meshgrid(x1,y1)
    f = interpolate.interp2d(x,y,Z,kind='cubic')
    Z1 = f(x1,y1)


    # 求拟合后最大值
    # print(np.unravel_index(np.argmax(Z1),Z1.shape))
    # x_index,y_index = np.unravel_index(np.argmax(Z1),Z1.shape)
    # print(x1[y_index],y1[x_index])

    # fig1 = plt.figure(figsize=(9, 8))
    # ax1 = fig1.add_subplot(121, projection='3d')
    #
    # ax1.set_xlabel('x-横向坐标')
    # ax1.set_ylabel('y-纵向坐标')
    # ax1.set_zlabel('z')
    # ax1.set_title('海水深度-坐标')
    #
    #
    # ax1.set_box_aspect((4000,5000,100))
    #
    #
    # surf1 = ax1.plot_surface(X, Y, Z, cmap=plt.cm.viridis)
    # ax2 = fig1.add_subplot(122, projection='3d')
    # surf2 = ax2.plot_surface(X1, Y1, Z1, cmap=plt.cm.viridis)
    # plt.show()
    return f

f = draw_fit_plane()
# 南北2海里,即2b
breadth = 1852 * 2                     #被我修改测试
# 东西4海里即2l
a = 1852*2.5


beta_list =[90]
num_list = [i for i in range(108,110,1)]


df = pd.read_excel('./附件.xlsx',header = 1,index_col=1)
df.drop('Unnamed: 0',axis = 1,inplace = True)
x = 1852*np.arange(0,4.01,0.02)
y = 1852*np.arange(0,5.01,0.02)
Z = np.array(df.values)
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
    # print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (X[0][0], X[1][0], X[2][0]))
    return X[0][0],X[1][0],X[2][0]
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


def calc_angle(v1,v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    dot_product = np.dot(v1,v2)
    cos_angle = dot_product / (norm1*norm2)
    # 注意是rad值
    angle = np.arccos(cos_angle)
    return angle
# 平面的参数parameter
Para = fit_plane(x,y,Z)
# 法向量
normal_vector = np.array([-Para[0],-Para[1],1])
angle = calc_angle(normal_vector,np.array([0,0,1]))

# 多波束换能器的开角
open_angle = np.deg2rad(120)
alpha = np.deg2rad(0.5953139400457111)
# 海域中心点处的海水深度
D0 = Para[0] * x[len(x)//2] + Para[1] * y[len(y)//2] + Para[2]

values = list()
# 计算最佳测线条数
groups = np.array([[waves_group(theta,num) for theta in beta_list] for num in num_list])
for i in groups:
    j = i[0]
    values = j.calc()
    # 如果找到了那么退出
    if len(values)!=0:
        break

# 计算汪洋哥坐标系的x，为后面转化为y做准备
calc_x_list = [float(i.value) for i in values]
def calc_y(calc_x_list):
    # 海里到米的系数
    coefficient = 1852
    y = list()
    for i in range(len(calc_x_list)):
        # if i== 0:
        #     y.append(2.5 * coefficient - calc_x_list[0])
        y.append(2.5 * coefficient - sum(calc_x_list[0:i+1]))
    return y

y_list = calc_y(calc_x_list)
# print(y_list)

def calc_ave_d(y,f):
    x_list = 1852*np.linspace(0,4,2000)
    values = np.array([f(x,y) for x in x_list])
    return np.average(values)

d_list = [calc_ave_d(y,f) for y in y_list]

def draw_rectangle(y_list,d_list):
    def calc_over_lap(y_list1,y_list2):
        length = len(y_list1)
        overlap_pos = list()
        overlap_neg = list()
        for i in range(length - 1):
            diff = -(y_list1[i] - y_list2[i+1])
            ratio = diff/max((y_list2[i] - y_list1[i]), (y_list2[i+1] - y_list1[i+1]))
            if diff < 0:
                overlap_neg.append(diff)
            if ratio > 0.2:
                overlap_pos.append(ratio)
        return overlap_pos,-np.array(overlap_neg)
    def calc_neg_ratio(overlap_neg,b):
        return sum(overlap_neg)/b

    a = 1852 * 4
    b = 1852 * 5
    y_list1 = [y_list[i] - d_list[i] * 1.6 **0.5 for i in range(len(y_list))]
    y_list2 = [y_list[i] + d_list[i] * 1.6**0.5 for i in range(len(y_list))]
    overlap_pos,overlap_neg= calc_over_lap(y_list1,y_list2)
    print(d_list)
    print((overlap_pos))
    print(calc_neg_ratio(overlap_neg,b))
    '''
    # 创建坐标系
    fig, ax = plt.subplots()

    # 设置x轴和y轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    for line1y,line2y in zip(y_list1,y_list2):
        ax.plot([0, a], [line1y]*2, color='blue')
        ax.plot([0, a], [line2y]*2, color='red')


    # 绘制矩形
    rectangle = patches.Rectangle((0, 0), a, b, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rectangle)
    ax.scatter(a,b)
    plt.show()
    '''

draw_rectangle(y_list,d_list)


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
