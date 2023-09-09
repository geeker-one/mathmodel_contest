import numpy as np

# 变量定义
# 两测线间距
d = 200
# 坡度
alpha = np.deg2rad(1.5)
# 多波束换能器的开角
theta = np.deg2rad(120)
# 中心深度
D0 = 70

class single_wave:
    def __init__(self,center_distance:float):
        self.center_distance = center_distance
        self.depth = D0 - center_distance * np.tan(alpha)
        self.cover_width = self.depth * np.sin(theta/2) * (1/ np.cos(theta/2 + alpha) +1/ np.cos(theta/2 -alpha))
        self.overlap_rate = 0.

def calc_overlap_rate(n:single_wave,n_plus1:single_wave):
    # 计算重合长度
    l_n_plus1 = n.depth * np.sin(theta/2) / np.cos(theta/2-alpha) + n_plus1.depth*np.sin(theta/2) / np.cos(theta/2+alpha) - d / np.cos(alpha)
    # 如果计算出的重合长度小于0则设为0
    if l_n_plus1 < 0:
        l_n_plus1 = 0
    n_plus1.overlap_rate = l_n_plus1 / n_plus1.cover_width

# 创建多个测线对象
waves = [single_wave(i) for i in range(-800,801,200)]

# 计算重叠率
for i in range(len(waves)-1):
    calc_overlap_rate(waves[i],waves[i+1])

# 打印结果
for i in waves:
    print(f"测线距中心点处距离{i.center_distance}m,海水深度{i.depth}m,覆盖宽度{i.cover_width}m,重叠率{i.overlap_rate * 100}%")
