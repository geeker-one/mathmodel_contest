import numpy as np
import cvxpy as cp


# 南北2海里,即2b
breadth = 1852 * 1
# 东西4海里即2l
a = 1852*2

# 多波束换能器的开角
open_angle = np.deg2rad(120)
alpha = np.deg2rad(1.5)
# 海域中心点处的海水深度
D0 = 110

beta_list =[90]
num_list = [i for i in range(5,100,1)]

class waves_group:
    def __init__(self,theta,num) -> None:
        self.degtheta = theta
        self.theta = np.deg2rad(theta)
        self.num = num
        self.x = [cp.Variable(nonneg=True) for _ in range(num)]
        self.alpha_ = np.arctan(np.tan(alpha) * np.cos(np.pi / 2 - self.theta))
        self.cons = self.get_cons()
        self.obj = cp.Minimize(self.func())
        print("初始化成功...")
        self.calc()
    def calc(self):
        prob = cp.Problem(self.obj,self.cons)
        prob.solve(solver = 'GLPK_MI')
        print(f"角度为{self.degtheta},num为{self.num}:{prob.status}")
        print(f"结果:{prob.value}")
        print(f"最优解：",end='')
        for i in self.x:
            print(i,end=',')
    def func(self):
        return 2*breadth * self.num/np.sin(self.theta)
    def get_cons(self):
        cons = [
            a * np.sin(self.theta) + breadth*np.cos(self.theta)+self.x[0] >=0,
            a * np.sin(self.theta) - breadth*np.cos(self.theta)+self.x[0] >=0,
            -a * np.sin(self.theta) - breadth*np.cos(self.theta)+sum(self.x) <=0,
            -a * np.sin(self.theta) + breadth*np.cos(self.theta)+sum(self.x) <=0,
            a * np.sin(self.theta) + breadth *np.cos(self.theta) + self.x[0] <= np.tan(self.theta/2)*(D0 + np.tan(alpha) * (a * np.cos(self.theta)**2 - breadth*np.sin(self.theta)*np.cos(self.theta) - self.x[0] * np.sin(self.theta))),
            a*np.sin(self.theta) + breadth *np.cos(self.theta) - sum(self.x) <= np.tan(self.theta/2) *(D0 + np.tan(alpha) * (breadth*np.sin(self.theta) * np.cos(self.theta) - a *np.cos(self.theta)**2 - np.sin(self.theta) * sum(self.x)))
        ]
        # 上面交点
        for i in range(2,self.num+1):
            x_i1 = (breadth* np.cos(self.theta) - sum(self.x[0:i])) / np.sin(self.theta)
            cons.append(
                2*np.tan(self.theta/2)*(D0 + x_i1 * np.tan(alpha)) + self.x[i-1] * np.tan(self.alpha_) * np.tan(self.theta/2) - self.x[i-1] >= 0.1 * 2*(D0+ x_i1 * np.tan(alpha)) * np.tan(self.theta/2)
            )
            cons.append(
                2*np.tan(self.theta/2)*(D0 + x_i1 * np.tan(alpha)) + self.x[i-1] * np.tan(self.alpha_) * np.tan(self.theta/2) - self.x[i-1] <= 0.2* 2*(D0+ x_i1 * np.tan(alpha)) * np.tan(self.theta/2)
            )

        # 下面交点
        for i in range(1,self.num):
            x_i2 = (-breadth * np.cos(self.theta) - sum(self.x[0:i]))/np.sin(self.theta)
            cons.append(
                2*np.tan(self.theta/2)*(D0 + x_i2 * np.tan(alpha)) - self.x[i] * np.tan(self.alpha_) * np.tan(self.theta/2) - self.x[i] >= 0.1 * 2*(D0+ x_i2 * np.tan(alpha) - self.x[i]*np.tan(self.alpha_)) * np.tan(self.theta/2)
            )
            cons.append(
                2*np.tan(self.theta/2)*(D0 + x_i2 * np.tan(alpha)) - self.x[i] * np.tan(self.alpha_) * np.tan(self.theta/2) - self.x[i] <= 0.2 * 2*(D0+ x_i2 * np.tan(alpha) - self.x[i]*np.tan(self.alpha_)) * np.tan(self.theta/2)
            )
        return cons


groups = np.array([[waves_group(theta,num) for theta in beta_list] for num in num_list])
for i in groups:
    for j in i:
        print(j.calc())
