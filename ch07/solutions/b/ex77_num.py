import numpy as np
N = np.array([100, 300, 1000.]); M = np.array([124.5, 101.1, 68.3])
b, a = np.polyfit(np.log(N), np.log(M), 1); print("slope", b, "a", a, "fit", np.exp(a) * N ** b)
# exact values from json
import json; r = json.load(open("../../../ch07/labs/77_results.json"))
Mx = np.array([r[f"schnet_N{n}"]["300K"]["F_mae"] * 1e3 for n in (100, 300, 1000)]); print("json", Mx, np.polyfit(np.log(N), np.log(Mx), 1)[0])
Mn = np.array([r[f"nequip_N{n}"]["300K"]["F_mae"] * 1e3 for n in (100, 300, 1000)]); print("nequip", Mn, np.polyfit(np.log(N), np.log(Mn), 1)[0])
N34 = np.exp((np.log(34) - a) / b); print("N for 34:", N34, " from N=1000 anchor:", 1000 * (34 / 68.3) ** (1 / b))
# 2-point slopes
print("2pt slopes", np.log(M[1]/M[0])/np.log(3), np.log(M[2]/M[1])/np.log(1000/300))
# Lab 7.9 schnet
N9 = np.array([100,300,1000.]); M9=np.array([64.8,58.9,63.4]); print("Lab7.9 slope", np.polyfit(np.log(N9), np.log(M9),1)[0])
# 7.7.6
d = np.radians(10); copies = (2 * np.pi / d) ** 3; print("copies (2pi/delta)^3", copies, " SO(3) volume 8pi^2/delta^3:", 8*np.pi**2/d**3)
print("total for N=300:", 300 * copies, " for N=1000:", 1000*copies)
print("aug samples 4000*32 =", 4000*32, " per structure (N=1000):", 4000*32/1000, " -> effective delta for 128 rotations: ", np.degrees(2*np.pi/128**(1/3)), "deg; with 8pi^2:", np.degrees((8*np.pi**2/128)**(1/3)))
print("x6 (S3) ->", 300*copies*6, " x3 (C3) ->", 300*copies*3)
print("full permutation group of ethanol H's: methyl 3! x methylene 2! =", 6*2, "->", 300*copies*12)
