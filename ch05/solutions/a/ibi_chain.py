"""Helper for Ex 5.4.4-5.4.6: reproduce the Lab IBI chain (same functions, seeds, settings) and cache it.
Run directly to (re)build the cache; other scripts call get_chain()."""
import os, sys, importlib, time
import numpy as np
sys.path.insert(0, "../ch05/labs")
ibi = importlib.import_module("54_ibi")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ibi_chain_cache.npz")
NMOL, T, DT = ibi.NMOL, ibi.T, 0.005
L = np.sqrt(NMOL / ibi.RHO)

def ibi_iterations(g_t, Lbox, alpha, nit, seed0, verbose=True):
    """IBI exactly as in the Lab (finalize after each update); returns potentials, g's, RMSE, P."""
    valid = g_t > 1e-3
    V = np.zeros(ibi.NBIN); V[valid] = -T * np.log(g_t[valid])
    V, Ft = ibi.finalize(V, valid)
    Vs, gs, errs, Ps = [V.copy()], [], [], []
    for it in range(nit):
        hc, nf, Pcg, _ = ibi.run_cg(NMOL, Lbox, Ft, 5000, 30000, DT, 1.0, seed0 + it, 15, 2.0)
        g = ibi.gofr(hc, nf, NMOL, Lbox)
        gs.append(g); errs.append(ibi.gerr(g, g_t)); Ps.append(Pcg)
        if verbose:
            print(f"    alpha={alpha} it={it}: g RMSE={errs[-1]:.4f}  P_CG={Pcg:.4f}", flush=True)
        if it == nit - 1:
            break
        m = valid & (g > 1e-3)
        V[m] += alpha * T * np.log(g[m] / g_t[m])
        V, Ft = ibi.finalize(V, valid)
        Vs.append(V.copy())
    return dict(V=V, Ft=Ft, Vs=np.array(Vs), gs=np.array(gs), errs=np.array(errs), Ps=np.array(Ps), valid=valid)

def get_chain():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    t0 = time.time()
    h, nfr, P_fine, _ = ibi.run_fine(NMOL, L, 15000, 80000, DT, 1.0, 1, 40)
    g_t = ibi.gofr(h, nfr, NMOL, L)
    print(f"  chain: fine MD P_fine={P_fine:.4f} ({time.time()-t0:.0f} s)", flush=True)
    out = ibi_iterations(g_t, L, 1.0, 9, 100)
    # pressure-correction step 0 (seed 300), exactly as the Lab
    hc, nf, P0, _ = ibi.run_cg(NMOL, L, out["Ft"], 5000, 30000, DT, 1.0, 300, 15, 2.0)
    g0 = ibi.gofr(hc, nf, NMOL, L)
    print(f"  chain: pressure step 0 P_CG={P0:.4f} ({time.time()-t0:.0f} s)", flush=True)
    res = dict(g_t=g_t, P_fine=P_fine, V=out["V"], Ft=out["Ft"], valid=out["valid"], Vs=out["Vs"],
               gs=out["gs"], errs=out["errs"], Ps=out["Ps"], g_p0=g0, P_p0=P0)
    np.savez(CACHE, **res)
    return res

if __name__ == "__main__":
    if os.path.exists(CACHE):
        os.remove(CACHE)
    c = get_chain()
    print("errs:", np.round(c["errs"], 4), "Ps:", np.round(c["Ps"], 4), "P_p0:", round(float(c["P_p0"]), 4))
