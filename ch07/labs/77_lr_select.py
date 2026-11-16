"""Ch.7.7: learning-rate choice for the two GNNs (N = 300, 1200 steps, batch 8). The printed list is the
VALIDATION force MAE (frames 2000-2099) during training; the dict is the test-set result. Usage: python 77_lr_select.py nequip 1e-2"""
import importlib.util, time, torch, os, numpy as np, sys
torch.set_num_threads(1)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sp=importlib.util.spec_from_file_location("m","77_models.py"); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
R,E,F,Z=m.load(300)
perm = np.random.default_rng(0).permutation(2000); tr=torch.as_tensor(perm[:300])
kind=sys.argv[1]; lr=float(sys.argv[2])
log=[]; t=time.process_time()
model=m.train(kind,R[tr],E[tr],F[tr],R[2000:2100],E[2000:2100],F[2000:2100],Z,steps=1200,bs=8,lr=lr,log=log)
print(kind,lr,[ (a,round(c,3)) for a,b,c in log[::4]], m.evaluate(model,R[2200:3000],E[2200:3000],F[2200:3000]), "%.0fs"%(time.process_time()-t),flush=True)
