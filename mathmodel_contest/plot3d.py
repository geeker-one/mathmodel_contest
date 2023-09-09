from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams["font.sans-serif"]=["Source Han Serif CN"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

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

fig1 = plt.figure(figsize=(9, 8))
ax1 = fig1.add_subplot(121, projection='3d')

ax1.set_xlabel('x-横向坐标')
ax1.set_ylabel('y-纵向坐标')
ax1.set_zlabel('z')
ax1.set_title('海水深度-坐标')


ax1.set_box_aspect((4000,5000,100))


surf1 = ax1.plot_surface(X, Y, Z, cmap=plt.cm.viridis)
# ax2 = fig1.add_subplot(122, projection='3d')
# surf2 = ax2.plot_surface(X1, Y1, Z1, cmap=plt.cm.viridis)
plt.show()
