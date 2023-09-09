import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"]=["Source Han Serif CN"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

#data = pd.read_csv('./cover_width.csv')
#data = data.to_numpy()
data = np.loadtxt('./cover_width.csv',delimiter=',')
print(list(data))
exit()

s = [i for i in np.arange(0,2.11,0.3)]
theta = [np.deg2rad(beta) for beta in np.arange(0,315+1,45)]


fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(111,polar = True)
mesh = ax.pcolormesh(theta,s,data,cmap = 'hot')

# for i in range(8):
#     for j in range(8):
#         plt.annotate(data[i][j],xy = (theta[i],s[j]),xytext = (5,2),textcoords='offset points', ha='right', va='bottom')

ax.plot()
fig.colorbar(mesh)

plt.title("极坐标力图")
plt.tight_layout()
plt.show()

#plt.show()
# print(data.columns)
# plt.show()
