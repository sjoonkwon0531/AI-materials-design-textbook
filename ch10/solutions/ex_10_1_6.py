"""Ex 10.1.6: per-user share of a pretraining cost C_pre = 1e6 eight-atom DFT calls, M users; break-even."""
import json
r = json.load(open('../data/101_cost.json'))
t8 = r['dft']['8']['cpu']; pd = r['summary']['slope_dft']; cml64 = r['mlip']['64']
Cpre = 1e6 * t8; Cft = 100.0; K = 30
own = 400 * t8 + 100.0
print(f"C_pre = {Cpre:.4g} s = {Cpre/3.156e7:.3g} core-years; own 400-label build = {own:.4g} s")
c64p, c64c = t8 * 8 ** pd, t8 * 8 ** 3
for M in [10, 1000, 1e5]:
    share = Cpre / M
    user = share + K * t8 + Cft
    print(f"M={M:>7g}: share {share:.4g} s = {share/t8:.4g} eight-atom DFT calls = {share/own:.3g} x own build; "
          f"user total (K={K}) {user:.4g} s; S*(N=64) = {user/(c64p-cml64):.3g} (slope) .. {user/(c64c-cml64):.3g} (cubic) steps")
print(f"user who ignores C_pre (K={K}): {(K*t8+Cft):.4g} s; S*(64) = {(K*t8+Cft)/(c64p-cml64):.3g} .. {(K*t8+Cft)/(c64c-cml64):.3g}")
for M in [10, 1000]:
    soc_pre = Cpre + M * (K * t8 + Cft); soc_own = M * own
    print(f"society, M={M}: pretrain route {soc_pre:.4g} s vs every group builds its own {soc_own:.4g} s; ratio {soc_pre/soc_own:.3g}")
print(f"M at which pretrain route equals own builds: {Cpre/(own-(K*t8+Cft)):.4g}")
