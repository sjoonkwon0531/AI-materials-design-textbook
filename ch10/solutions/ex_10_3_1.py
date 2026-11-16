"""Ex 10.3.1: ghost-shell thickness, volume and atom counts for a 30 A cubic domain, Si PBE density."""
rho = 1 / 20.453; L = 30.0
own = L ** 3 * rho
print(f"domain volume {L**3:.0f} A^3, atoms {own:.1f}")
for tag, d in [('strict local, d = 5 A', 5.0), ('5-layer MP, d = 25 A', 25.0)]:
    Vg = (L + 2 * d) ** 3 - L ** 3
    print(f"{tag}: ghost volume {Vg:.0f} A^3, ghost atoms {Vg*rho:.1f}, ghost/own {Vg/L**3:.3f}")
