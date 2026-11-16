import numpy as np
def u(x):  return np.exp(np.sin(x))
def du(x): return np.cos(x)*np.exp(np.sin(x))

def spectral_deriv(N):                       # 세 줄짜리 마법
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    k = np.fft.fftfreq(N, d=1.0/N)           # 정수 파수
    dp = np.real(np.fft.ifft(1j*k*np.fft.fft(u(x))))
    return np.max(np.abs(dp - du(x)))

def fdm_deriv(N):                            # 2차 중심차분, 주기
    x = np.linspace(0, 2*np.pi, N, endpoint=False); h = 2*np.pi/N
    uh = u(x); dp = (np.roll(uh,-1) - np.roll(uh,1))/(2*h)
    return np.max(np.abs(dp - du(x)))
