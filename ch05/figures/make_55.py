import sys, json; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'common'))
from bookstyle import *
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
L = '../ch05/labs/'; F = '../ch05/figures/'
d = json.load(open(L+'55_homog.json'))
imgs = np.load(L+'55_homog_imgs.npz')
two = ListedColormap(['#F2F2F2', NAVY])

def bounds_cond(f2, k1=1., k2=10.):
    f1 = 1-f2; V = f1*k1+f2*k2; R = 1/(f1/k1+f2/k2)
    HSu = k2 + f1/(1/(k1-k2) + f2/(2*k2)); HSl = k1 + f2/(1/(k2-k1) + f1/(2*k1)); return V, R, HSl, HSu
def lame(E, nu): return E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
def bounds_bulk(f2):
    l1, G1 = lame(1., .3); l2, G2 = lame(10., .3); K1, K2 = l1+G1, l2+G2; f1 = 1-f2
    V = f1*K1+f2*K2; R = 1/(f1/K1+f2/K2)
    HSu = K2 + f1/(1/(K1-K2) + f2/(K2+G2)); HSl = K1 + f2/(1/(K2-K1) + f1/(K1+G1)); return V, R, HSl, HSu

# ---------------- Fig 5.5.1
fig = plt.figure(figsize=(17.5, 4.8))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1.08, 0.18, 1.5], wspace=0.3)
ax = fig.add_subplot(gs[0]); ax.imshow(imgs['m-0.45'].T, origin='lower', cmap=two, extent=[0,1,0,1])
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xlabel('x'); ax.set_ylabel('y'); panel_label(ax, 'a', x=-0.2)
ax = fig.add_subplot(gs[1]); ax.imshow(imgs['m+0.00'].T, origin='lower', cmap=two, extent=[0,1,0,1])
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xlabel('x'); panel_label(ax, 'b', x=-0.2)
fl = np.load(L+'55_flux.npz'); n = 128
J = np.zeros((n, n)); J[fl['pix0'], fl['pix1']] = np.hypot(fl['jx'], fl['jy'])
ax = fig.add_subplot(gs[2]); im = ax.imshow(J.T, origin='lower', cmap='magma', norm=LogNorm(0.1, 30), extent=[0,1,0,1])
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xlabel('x'); panel_label(ax, 'c', x=-0.2)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r'$|\mathbf{j}|$')
ax = fig.add_subplot(gs[4])
f = np.linspace(0, 1, 201); V, R, HSl, HSu = bounds_cond(f)
ax.fill_between(f, HSl, HSu, color=ICE, alpha=0.9, lw=0, label='HS band')
ax.plot(f, V, '-', color=RED, lw=2, label='Voigt'); ax.plot(f, R, '--', color=BLUE, lw=2, label='Reuss')
ax.plot(f, HSu, ':', color=NAVY, lw=1.6); ax.plot(f, HSl, ':', color=NAVY, lw=1.6)
rows = np.array(d['cond_rows'])
ax.plot(rows[:,0], rows[:,1], 'o', color=GOLD, mec='k', ms=9, label='FEM (CH maps)')
kis = np.array(d['kis'])
ax.errorbar([0.5], [kis.mean()], yerr=[kis.std(ddof=1)/np.sqrt(len(kis))], fmt='s', color=GREEN, mec='k', ms=9, capsize=4, label='8-seed mean')
ax.axhline(np.sqrt(10), color=GRAY, lw=1, ls='-.'); ax.text(0.02, np.sqrt(10)+0.15, r'$\sqrt{k_1k_2}$', color=GRAY, fontsize=15)
ax.set_xlabel(r'volume fraction $f_2$'); ax.set_ylabel(r'$k_{\rm eff}/k_1$'); ax.set_xlim(0, 1); ax.set_ylim(0, 10.5)
ax.legend(loc='upper left', frameon=False, fontsize=13, bbox_to_anchor=(0.0, 0.93)); panel_label(ax, 'd', x=-0.13)
fig.savefig(F+'Fig5.5.1_homogenization.png'); plt.close(fig)

# ---------------- Fig 5.5.2
fig = plt.figure(figsize=(16.5, 4.8))
gs = fig.add_gridspec(1, 4, width_ratios=[1.12, 0.22, 1.3, 1.1], wspace=0.38)
sxx = np.load(L+'55_sxx.npy'); S = np.zeros((n, n)); S[fl['pix0'], fl['pix1']] = sxx
ax = fig.add_subplot(gs[0]); im = ax.imshow(S.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=np.percentile(S, 99.5), extent=[0,1,0,1])
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xlabel('x'); ax.set_ylabel('y'); panel_label(ax, 'a', x=-0.2)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r'$\sigma_{xx}$ ($\bar\varepsilon_{xx}=1$)')
ax = fig.add_subplot(gs[2]); V, R, HSl, HSu = bounds_bulk(f)
ax.fill_between(f, HSl, HSu, color=ICE, alpha=0.9, lw=0, label='HS band')
ax.plot(f, V, '-', color=RED, lw=2, label='Voigt'); ax.plot(f, R, '--', color=BLUE, lw=2, label='Reuss')
el = np.array(d['el_rows']); ax.plot(el[:,0], el[:,1], 'o', color=GOLD, mec='k', ms=9, label='FEM')
ax.set_xlabel(r'volume fraction $f_2$'); ax.set_ylabel(r'$\kappa_{\rm eff}$ ($E_1$ = 1)'); ax.set_xlim(0, 1)
ax.legend(loc='upper left', frameon=False, fontsize=13); panel_label(ax, 'b', x=-0.15)
ax = fig.add_subplot(gs[3]); ref = np.array(d['refine'])
ax.plot(ref[:,1], ref[:,2], 'o-', color=NAVY, ms=9, lw=2)
ax.set_xscale('log'); ax.set_xlabel('number of elements'); ax.set_ylabel(r'$k_{xx}^{h}$')
ax.set_xticks([1e3, 1e4, 1e5]); panel_label(ax, 'c', x=-0.24)
fig.savefig(F+'Fig5.5.2_elastic_bounds.png'); plt.close(fig)

# ---------------- Fig 5.5.3  (LBM)
fig = plt.figure(figsize=(16, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.42, wspace=0.34)
p = np.load(L+'55_lbm_prof.npz')
ax = fig.add_subplot(gs[0, 0]); yy = np.linspace(0, 16, 200); nu = 0.1; g = 0.01*8*nu/16**2
ax.plot(g/(2*nu)*yy*(16-yy)*100, yy, '-', color=NAVY, lw=2, label='analytic')
ax.plot(p['u']*100, p['y'], 'o', color=RED, ms=7, label=r'LBM, $\tau$=0.8')
ax.set_xlabel(r'$u_x$ ($10^{-2}$ lattice units)'); ax.set_ylabel('y (lattice units)'); ax.set_ylim(0, 16)
ax.legend(frameon=False, loc='lower left', fontsize=13); panel_label(ax, 'a', x=-0.2)
tt = np.load(L+'55_lbm_tau.npy'); ax = fig.add_subplot(gs[0, 1])
ax.semilogy(tt[:,0], tt[:,1], 'o-', color=NAVY, ms=8, lw=2)
ax.axvline(0.5+np.sqrt(3/16), color=RED, ls='--', lw=1.5); ax.text(0.95, 2e-6, r'$\tau=\frac{1}{2}+\sqrt{3/16}$', color=RED, fontsize=15)
ax.set_xlabel(r'relaxation time $\tau$'); ax.set_ylabel(r'relative $L^2$ error'); panel_label(ax, 'b', x=-0.2)
nn = np.load(L+'55_lbm_n.npy'); ax = fig.add_subplot(gs[0, 2])
ax.loglog(nn[:,0], nn[:,1], 's-', color=NAVY, ms=8, lw=2, label=r'LBM, $\tau$=0.8')
ax.loglog(nn[:,0], nn[0,1]*(nn[:,0]/nn[0,0])**-2, ':', color=GRAY, lw=2, label='slope −2')
ax.set_xlabel(r'channel width $n_y$'); ax.set_ylabel(r'relative $L^2$ error'); ax.legend(frameon=False, fontsize=13)
ax.set_xticks([8, 16, 32, 64]); ax.set_xticklabels(['8', '16', '32', '64']); panel_label(ax, 'c', x=-0.2)
for k, tag in enumerate(['lowRe', 'highRe']):
    z = np.load(L+f'55_lbm_cyl_{tag}.npz'); ux, uy, sol = z['ux'], z['uy'], z['solid']
    ax = fig.add_subplot(gs[1, :]) if False else fig.add_subplot(gs[1, 0:2] if k == 0 else gs[1, 2])
    pass
plt.close(fig)

# redo layout for LBM: 2 rows -> row1 three panels, row2 two cylinder panels
fig = plt.figure(figsize=(16, 9.4))
top = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.97, bottom=0.57, wspace=0.34)
bot = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.45, bottom=0.05, wspace=0.12)
ax = fig.add_subplot(top[0]); ax.plot(g/(2*nu)*yy*(16-yy)*100, yy, '-', color=NAVY, lw=2, label='analytic')
ax.plot(p['u']*100, p['y'], 'o', color=RED, ms=7, label=r'LBM, $\tau$=0.8')
ax.set_xlabel(r'$u_x$ ($10^{-2}$, lattice units)'); ax.set_ylabel('y (lattice units)'); ax.set_ylim(0, 16)
ax.legend(frameon=False, loc='lower left', fontsize=13); panel_label(ax, 'a', x=-0.2)
ax = fig.add_subplot(top[1]); ax.semilogy(tt[:,0], tt[:,1], 'o-', color=NAVY, ms=8, lw=2)
ax.axvline(0.5+np.sqrt(3/16), color=RED, ls='--', lw=1.5); ax.text(0.97, 1.5e-6, r'$\tau=1/2+\sqrt{3/16}$', color=RED, fontsize=15)
ax.set_xlabel(r'relaxation time $\tau$'); ax.set_ylabel(r'relative $L^2$ error'); panel_label(ax, 'b', x=-0.2)
ax = fig.add_subplot(top[2]); ax.loglog(nn[:,0], nn[:,1], 's-', color=NAVY, ms=8, lw=2, label=r'LBM, $\tau$=0.8')
ax.loglog(nn[:,0], nn[0,1]*(nn[:,0]/nn[0,0])**-2, ':', color=GRAY, lw=2, label='slope −2')
ax.set_xlabel(r'channel width $n_y$'); ax.set_ylabel(r'relative $L^2$ error'); ax.legend(frameon=False, fontsize=13)
ax.set_xticks([8, 16, 32, 64]); ax.set_xticklabels(['8', '16', '32', '64']); ax.minorticks_off(); panel_label(ax, 'c', x=-0.2)
for k, tag in enumerate(['lowRe', 'highRe']):
    z = np.load(L+f'55_lbm_cyl_{tag}.npz'); ux, uy, sol = z['ux'], z['uy'], z['solid']
    ax = fig.add_subplot(bot[k]); sp = np.hypot(ux, uy); sp = np.ma.masked_where(sol, sp/np.abs(ux[~sol]).mean())
    im = ax.imshow(sp.T, origin='lower', cmap='viridis', vmin=0, vmax=2.6, aspect='equal')
    X, Y = np.meshgrid(np.arange(ux.shape[0]), np.arange(ux.shape[1]))
    ax.streamplot(X, Y, ux.T, uy.T, color='w', density=1.3, linewidth=0.7, arrowsize=0.6)
    ax.set_xlim(0, ux.shape[0]-1); ax.set_ylim(0, ux.shape[1]-1); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.98, 0.9, f"Re = {float(z['Re']):.1f}", transform=ax.transAxes, ha='right', color='w', fontsize=17)
    panel_label(ax, 'de'[k], x=-0.04, y=1.03)
cax = fig.add_axes([0.985, 0.08, 0.008, 0.34]); cb = fig.colorbar(im, cax=cax); cb.set_label(r'$|u|/\langle u_x\rangle$', fontsize=15)
fig.savefig(F+'Fig5.5.3_lbm.png'); plt.close(fig)
print('done')
