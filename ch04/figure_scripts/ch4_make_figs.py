import sys, io, re, contextlib, runpy, numpy as np
sys.path.insert(0, '.')
import logging; logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
from bookstyle import *
LAB='../ch04/labs/'
def run(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(LAB+name, run_name='__main__')
def save(fig, name):
    fig.savefig(name); plt.close(fig); print('saved', name)
only = sys.argv[1:] or ['41','42','43','44','45','46','47']

if '41' in only:
    d = run('lab_4_1.py'); x=d['x']; E=d['E_ho']; psi=d['psi_ho']; dx=x[1]-x[0]
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    ax.plot(x, 0.5*x**2, color=GRAY, lw=1.6, ls='--', label='$V(x)=x^2/2$')
    for n,c in zip(range(4), [NAVY,RED,BLUE,GREEN]):
        p = psi[:,n]/np.sqrt(dx); p *= np.sign(p[np.argmax(np.abs(p))])
        ax.axhline(E[n], color=c, lw=0.7, alpha=0.5)
        ax.plot(x, E[n]+0.45*p, color=c, lw=2.2, label=f'$n={n}$, $E={E[n]:.3f}$')
    ax.set_xlim(-5,5); ax.set_ylim(0,5.9); ax.set_xlabel('$x$'); ax.set_ylabel(r'$E_n + \psi_n(x)$ (offset)')
    ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=13)
    save(fig,'Fig4.1.1_harmonic.png')
    Ns=np.array(d['Ns']); errs=np.array(d['errs'])
    fig, ax = plt.subplots(figsize=(6.8,5.2))
    ax.loglog(Ns, errs, 'o-', color=NAVY, lw=2.2, ms=8, label='FDM ground-state error')
    ax.loglog(Ns, errs[0]*(Ns/Ns[0])**-2.0, '--', color=RED, lw=1.6, label='slope $-2$')
    ax.set_xlabel('number of grid points $N$'); ax.set_ylabel(r'$|E_0^{\rm FDM}-0.5|$'); ax.legend(frameon=False)
    save(fig,'Fig4.1.2_convergence.png')
    a=d['avals']; Ev=d['Evar']
    fig, ax = plt.subplots(figsize=(6.8,5.2))
    ax.plot(a, Ev, color=NAVY, lw=2.4, label=r'$\langle \hat H\rangle(a)=a/4+1/(4a)$')
    ax.axhline(0.5, color=RED, ls='--', lw=1.6, label='exact $E_0=0.5$')
    ax.plot([1.0],[0.5],'o',color=GOLD,ms=12,mec=NAVY,zorder=5,label='minimum, $a=1$')
    ax.set_xlabel('trial width parameter $a$'); ax.set_ylabel('trial energy'); ax.set_ylim(0.4,1.4); ax.legend(frameon=False)
    save(fig,'Fig4.1.3_variational.png')

if '42' in only:
    d = run('lab_4_2.py'); scf=d['scf']; x=d['x']; Vext=d['Vext']; pot=d['potentials']
    Eh,dnh,n,eps,psi = scf(lamH=0.5, alpha=0.3)
    vH,vx = pot(n,0.5); veff=Vext+vH+vx
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    ax.plot(x, Vext, '--', color=GRAY, lw=1.8, label=r'$v_{\rm ext}$')
    ax.plot(x, veff, color=NAVY, lw=2.4, label=r'$v_{\rm eff}=v_{\rm ext}+v_H+v_{xc}$')
    ax.set_xlim(-5,5); ax.set_ylim(0,8); ax.set_xlabel('$x$'); ax.set_ylabel('potential')
    ax2 = ax.twinx(); ax2.fill_between(x, n, color=GOLD, alpha=0.45, label='$n(x)$'); ax2.set_ylabel('density $n(x)$')
    ax2.spines['right'].set_visible(True); ax2.set_ylim(0, n.max()*1.9)
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,loc='upper center',frameon=False,fontsize=14)
    save(fig,'Fig4.2.1_scf_solution.png')
    fig, ax = plt.subplots(figsize=(6.8,5.2))
    ax.plot(np.arange(len(Eh)), Eh, 'o-', color=NAVY, ms=4, lw=1.8)
    ax.axhline(Eh[-1], color=RED, ls='--', lw=1.4, label=f'converged $E={Eh[-1]:.4f}$')
    ax.set_xlabel('SCF iteration'); ax.set_ylabel('total energy'); ax.legend(frameon=False)
    save(fig,'Fig4.2.2_scf_energy.png')
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    for al,c,ls in zip([1.0,0.5,0.4,0.3,0.15],[RED,PURPLE,NAVY,BLUE,GOLD],['-','--','-','-','-']):
        _,dn,*_ = scf(lamH=4.0, alpha=al, n_iter=200, tol=0.0)
        ax.semilogy(np.arange(1,len(dn)+1), dn, color=c, ls=ls, lw=2.0, label=rf'$\alpha={al}$')
    ax.axhline(1e-9, color=GRAY, ls=':', lw=1.2)
    ax.set_xlabel('SCF iteration'); ax.set_ylabel(r'$\|\Delta n\|$'); ax.legend(frameon=False, ncol=2)
    save(fig,'Fig4.2.3_mixing.png')

if '43' in only:
    txt=open(LAB+'out_4_3.txt').read()
    rows=re.findall(r'^\s+(\d+)\s+([\d.]+)\s+\d+\s*$', txt, re.M)
    N=np.array([float(r[0]) for r in rows]); t=np.array([float(r[1]) for r in rows])
    m=N>=800; p,c=np.polyfit(np.log(N[m]),np.log(t[m]),1)
    fig, ax = plt.subplots(figsize=(6.8,5.2))
    ax.loglog(N,t,'o-',color=NAVY,lw=2.2,ms=8,label='measured (single thread)')
    Nf=np.array([600,3600]); ax.loglog(Nf, np.exp(c)*Nf**p, '-', color=GOLD, lw=2.0, label=f'tail fit, $p={p:.2f}$')
    ax.loglog(Nf, t[m][0]*(Nf/N[m][0])**3, '--', color=RED, lw=1.5, label='slope 3')
    ax.set_xlabel('matrix size $N$'); ax.set_ylabel('wall time (s)'); ax.legend(frameon=False)
    save(fig,'Fig4.3.1_scaling.png')

if '44' in only:
    d = run('lab_4_4.py')
    fig,(a,b)=plt.subplots(1,2,figsize=(13,5.4))
    for X,V,c,lab in [(d['Xe'],d['Ve'],RED,'forward Euler'),(d['Xs'],d['Vs'],GREEN,'symplectic Euler'),(d['Xv'],d['Vv'],NAVY,'velocity Verlet')]:
        a.plot(X,V,color=c,lw=1.6,label=lab)
    a.set_xlim(-4,4); a.set_ylim(-4,4); a.set_aspect('equal'); a.set_xlabel('$x$'); a.set_ylabel('$v$')
    st=np.arange(len(d['Ev']))
    for E,c,lab in [(d['Ee'],RED,'forward Euler'),(d['Es'],GREEN,'symplectic Euler'),(d['Ev'],NAVY,'velocity Verlet')]:
        b.semilogy(st,E,color=c,lw=1.8,label=lab)
    b.axhline(0.5,color=GRAY,ls=':',lw=1.2); b.set_xlabel('step'); b.set_ylabel('energy')
    h,l=b.get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.07))
    panel_label(a,'a'); panel_label(b,'b',x=-0.16)
    fig.tight_layout(); save(fig,'Fig4.4.1_symplectic.png')
    nve=d['nve']; nvt=d['nvt']
    fig, ax = plt.subplots(figsize=(7.6,5.2))
    ax.plot(nve['t'],nve['KE'],color=GREEN,lw=1.2,label='kinetic')
    ax.plot(nve['t'],nve['PE'],color=PURPLE,lw=1.2,label='potential')
    ax.plot(nve['t'],nve['E'],color=NAVY,lw=2.4,label='total')
    ax.set_xlabel(r'time ($\tau_{\rm LJ}$)'); ax.set_ylabel(r'energy ($\varepsilon$)'); ax.legend(frameon=False,ncol=3,loc='center right',fontsize=14)
    save(fig,'Fig4.4.2_lj_nve.png')
    fig, ax = plt.subplots(figsize=(7.6,5.2))
    ax.plot(nve['t'],nve['T'],color=GRAY,lw=1.2,label='NVE (no thermostat)')
    ax.plot(nvt['t'],nvt['T'],color=NAVY,lw=1.4,label='NVT (velocity rescaling)')
    ax.axhline(1.0,color=RED,ls='--',lw=1.6,label='target $T_0=1$')
    ax.set_ylim(0.5,1.85); ax.set_xlabel(r'time ($\tau_{\rm LJ}$)'); ax.set_ylabel('temperature (reduced)'); ax.legend(frameon=False,ncol=2,loc='upper center',fontsize=14)
    save(fig,'Fig4.4.3_thermostat.png')

if '45' in only:
    d = run('lab_4_5.py'); rg=d['rg']; m=d['model']; L=d['learn']
    fig,(a,b)=plt.subplots(1,2,figsize=(13,5.4))
    a.plot(rg,d['U_true'](rg),color=NAVY,lw=2.6,label='true LJ')
    a.plot(rg,m.U(rg),'--',color=RED,lw=2.0,label='learned (KRR)')
    a.plot(d['rt'],d['ut'],'o',color=GOLD,ms=4,alpha=0.7,label='training points')
    a.set_ylim(-1.5,2); a.set_xlabel(r'$r/\sigma$'); a.set_ylabel(r'$U(r)/\varepsilon$'); a.legend(frameon=False)
    ns=[l[0] for l in L]; b.loglog(ns,[l[1] for l in L],'o-',color=NAVY,lw=2,ms=8,label='energy RMSE')
    b.loglog(ns,[l[2] for l in L],'s-',color=RED,lw=2,ms=8,label='force RMSE')
    b.set_xlabel('number of training points'); b.set_ylabel('RMSE (LJ units)'); b.legend(frameon=False)
    panel_label(a,'a'); panel_label(b,'b')
    fig.tight_layout(); save(fig,'Fig4.5.1_learned_pes.png')
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    ax.plot(d['r_t'],d['g_t'],color=NAVY,lw=2.6,label='MD with true LJ')
    ax.plot(d['r_m'],d['g_m'],'--',color=RED,lw=2.0,label='MD with learned potential')
    ax.axhline(1,color=GRAY,ls=':',lw=1.2); ax.set_xlim(0,4); ax.set_xlabel(r'$r/\sigma$'); ax.set_ylabel('$g(r)$'); ax.legend(frameon=False)
    save(fig,'Fig4.5.2_gofr.png')

if '46' in only:
    z=np.load(LAB+'lab46_data.npz'); v=z['vols']; e=z['ens']
    from ase.eos import EquationOfState
    eos=EquationOfState(list(v),list(e)); v0,e0,B=eos.fit()
    vf=np.linspace(v.min(),v.max(),200); pc=np.polyfit(v**(-1/3),e,3); ef=np.polyval(pc,vf**(-1/3))   # SJ form: cubic in V^(-1/3)
    fig,(a,b)=plt.subplots(1,2,figsize=(13,5.2))
    a.plot(v,e,'o',color=NAVY,ms=8,label='EMT energies'); a.plot(vf,ef,'-',color=RED,lw=2,label='EOS fit (ASE default, SJ)')
    a.axvline(v0,color=GRAY,ls=':',lw=1.4)
    a.set_xlabel(r'volume (Å$^3$, 4-atom cell)'); a.set_ylabel('energy (eV)'); a.legend(frameon=False,fontsize=14)
    a.text(0.74,0.55,f'$a_0$ = {float(z["a"]):.3f} Å\n$B$ = {float(z["B"]):.0f} GPa',transform=a.transAxes,ha='center',fontsize=16)
    T=z['Ts']; t=np.arange(len(T))*10*2.0/1000
    b.plot(t,T,color=NAVY,lw=1.4,label='instantaneous $T$'); b.axhline(300,color=RED,ls='--',lw=1.6,label='target 300 K')
    b.set_xlabel('time (ps)'); b.set_ylabel('temperature (K)'); b.legend(frameon=False,fontsize=14)
    panel_label(a,'a',x=-0.2); panel_label(b,'b')
    fig.tight_layout(); save(fig,'Fig4.6.1_ase.png')

if '47' in only:
    z=np.load(LAB+'lab47_data.npz'); sites=[str(s) for s in z['sites']]; Ea=z['Eads']
    fig,(a,b,c)=plt.subplots(1,3,figsize=(16,5.0))
    cols=[GOLD if i==int(np.argmin(Ea)) else NAVY for i in range(len(sites))]
    a.bar(sites,Ea,color=cols,edgecolor=NAVY); a.axhline(0,color='k',lw=0.8)
    a.set_ylabel(r'$E_{\rm ads}$ (eV)'); a.set_ylim(-0.6,0)
    for b_,lab,xs,ys,col in [(b,'number of slab layers',z['lay'],z['Elay'],NAVY),(c,'vacuum thickness (Å)',z['vac'],z['Evac'],GREEN)]:
        b_.plot(xs,ys,'o-',color=col,lw=2,ms=9); b_.set_xlabel(lab); b_.set_ylabel(r'$E_{\rm ads}$ (eV)'); b_.set_ylim(-0.57,-0.47)
    panel_label(a,'a',x=-0.25); panel_label(b,'b',x=-0.25); panel_label(c,'c',x=-0.25)
    fig.tight_layout(); save(fig,'Fig4.7.1_adsorption.png')
    fig, ax = plt.subplots(figsize=(7.6,5.4))
    g=np.linspace(-0.8,0.8,300); ax.plot(g,-np.abs(g),color=NAVY,lw=2.6)
    ax.axvline(0,color=RED,ls='--',lw=1.6)
    ax.plot([-0.09],[-0.09],'o',color=GOLD,ms=13,mec=NAVY,zorder=5); ax.annotate('Pt ($\\Delta G_H\\approx-0.09$ eV)',(-0.09,-0.09),xytext=(0.06,-0.02),fontsize=15)
    ax.text(-0.52,-0.22,'binds H too strongly\n(e.g. Mo, W, Ni)\ndesorption-limited',ha='center',fontsize=14,color=GRAY)
    ax.text(0.52,-0.22,'binds H too weakly\n(e.g. Au, Ag)\nadsorption-limited',ha='center',fontsize=14,color=GRAY)
    ax.set_xlabel(r'$\Delta G_{\rm H}$ (eV)'); ax.set_ylabel(r'HER activity, $\log i_0$ (schematic)'); ax.set_yticks([]); ax.set_ylim(-0.9,0.08)
    save(fig,'Fig4.7.2_volcano.png')
