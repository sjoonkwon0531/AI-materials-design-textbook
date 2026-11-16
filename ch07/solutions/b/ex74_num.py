import numpy as np
# 7.4.1
c=2.99792458e10; amu=1.66053906660e-27; kB=1.380649e-23; hbar=1.054571817e-34
nu=2950; mu=0.92*amu; w=2*np.pi*c*nu; k=mu*w**2; T=300
s_cl=np.sqrt(kB*T/k)*1e10; s_zp=np.sqrt(hbar/(2*mu*w))*1e10
print(f"omega {w:.4e} rad/s, k {k:.1f} N/m = {k*1e-20/1.602176634e-19:.2f} eV/A^2")
print(f"classical sigma {s_cl:.4f} A, zero-point sigma {s_zp:.4f} A, ratio {s_zp/s_cl:.2f}")
x=hbar*w/(kB*T); print("hbar w / kT", x, "quantum thermal sigma", s_zp*np.sqrt(1/np.tanh(x/2)))
print("kT eV", kB*T/1.602176634e-19)
# energy of sigma noise in harmonic bond: 1/2 k sigma^2 for 0.05,0.02
kev=k*1e-20/1.602176634e-19
for s in (0.02,0.05): print(s, "0.5 k s^2 =", 0.5*kev*s**2, "eV; in kT", 0.5*kev*s**2/(kB*T/1.602176634e-19), " s/s_cl", s/s_cl)
# 7.4.4
D=np.array([1e3,1e4,1e5]); M=np.array([120,68,38.])
b,a=np.polyfit(np.log10(D),np.log10(M),1); alpha=-b; c0=10**a
print("alpha",alpha,"c",c0, "fit", c0*D**-alpha)
D30=(c0/30)**(1/alpha); print("D30",D30, "D30/3", D30/3)
a2=1.5*alpha; # keep anchor? two options: same c, or same MAE at D=1e3
D30b=(c0/30)**(1/a2); print("1.5alpha same c:", D30b)
c1=120*1e3**a2; D30c=(c1/30)**(1/a2); print("1.5alpha anchored at D=1e3 (120):", D30c, c1)
# anchored at D=1e4 (68)
c2=68*1e4**a2; print("anchored 1e4:", (c2/30)**(1/a2))
