import numpy as np
import pandas as pd

# 海域中心点处的海水深度
D0 = 120
deg_alpha = 1.5
alpha = np.deg2rad(deg_alpha)
theta = np.deg2rad(120)

class single_wave:
    def __init__(self,s:float,beta:float) -> None:
        self.s = s
        self.beta = np.deg2rad(beta)
        self.depth = self.calc_depth()
        self.cover_width = self.calc_cover_width()
    def calc_depth(self):
        phi = np.arctan(np.tan(alpha) * np.cos(self.beta))
        return D0 + self.s * np.tan(phi)
    def calc_cover_width(self):
        alpha_ = np.arctan(np.tan(alpha)*np.cos(np.pi/2 - self.beta))
        width = self.depth * np.sin(theta/2) * ( 1/ np.cos(theta/2+alpha_) + 1/np.cos(theta/2 - alpha_))
        return width

waves = np.array([[single_wave(s,beta).cover_width for beta in np.arange(0,315+1,45)] for s in 1825*np.arange(0,2.11,0.3)])
# waves = pd.DataFrame(waves)
# waves.index = [beta for beta in np.arange(0,315+1,45)]
# waves.columns = [s for s in np.arange(0,2.11,0.3)]
# waves.to_csv('./cover_width.csv')
np.savetxt('./cover_width.csv',waves.T,delimiter=',')
