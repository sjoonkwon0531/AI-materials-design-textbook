import numpy as np
def kepler(t, Y):
    x, y, vx, vy = Y; r3 = (x*x + y*y)**1.5
    return np.array([vx, vy, -x/r3, -y/r3])
Y0 = np.array([0.2, 0.0, 0.0, 3.0])         # e=0.8, a=1, perihelion r=0.2, speed 3, period 2*pi
T4 = 8*np.pi                                 # 4 periods; exact Y(T4) = Y0
energy = lambda Y: 0.5*(Y[2]**2 + Y[3]**2) - 1/np.hypot(Y[0], Y[1])
if __name__ == "__main__":
    e = 0.8; vp, va = np.sqrt((1+e)/(1-e)), np.sqrt((1-e)/(1+e))
    print(f"perihelion speed {vp:.4f}, aphelion speed {va:.4f}, ratio {vp/va:.2f}; E0={energy(Y0):.4f} (=-1/(2a)=-0.5)")
