import numpy as np
src = open("ex866_bbo_film.py").read().split("which = sys.argv")[0]
exec(src)
for rep in (1, 2):
    x, ls, nv = run_gp(1000, rep)
    print(f"  GP-EI budget 1000 rep {rep}: test J {test(x):.3f} | schedule " + " ".join(f"{T:.0f}" for T in E.to_T(x)) + f" (ls {ls:g}, noise {nv:g})", flush=True)
