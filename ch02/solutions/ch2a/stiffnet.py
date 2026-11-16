# Lab 2.3 stiff network: X1 -(1001.5)-> X2, X2 <-> X3 (kf=0.5, kr=1.0), X0=(1,0,0), 0<=t<=8
import numpy as np
k1, kf, kr = 1001.5, 0.5, 1.0
A = np.array([[-k1, 0.0, 0.0],
              [ k1, -kf,  kr],
              [0.0,  kf, -kr]])
X0 = np.array([1.0, 0.0, 0.0]); T = 8.0
