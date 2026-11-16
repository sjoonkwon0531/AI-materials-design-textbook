"""Exercise 3.6.C (network + MP API key required; NOT executed in the solution build:
the build sandbox blocks these hosts). Results depend on DB version and query date.
Usage: export MP_API_KEY=...; python3 sol_ch3_6C_network.py
"""
import datetime, json, requests
import matplotlib
import matplotlib.pyplot as plt

stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
prov = {"query_utc": stamp}

# (1) MP: stable Si-O phases and their (GGA/GGA+U or r2SCAN, per DB version) band gaps
from mp_api.client import MPRester
with MPRester() as mpr:
    prov["mp_db_version"] = mpr.db_version
    docs = mpr.materials.summary.search(
        chemsys="Si-O", is_stable=True, include_gnome=False,
        fields=["material_id", "formula_pretty", "band_gap", "energy_above_hull"])
rows = sorted(((d.formula_pretty, str(d.material_id), d.band_gap) for d in docs), key=lambda r: r[0])
for r in rows:
    print(f"{r[1]:>12s} {r[0]:<6s} band_gap = {r[2]:.3f} eV")

plt.rcParams["font.family"] = ["Arial Narrow", "DejaVu Sans"]
plt.rcParams["font.size"] = 14
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.bar([f"{f}\n{m}" for f, m, _ in rows], [g for *_, g in rows], color="#3b5b92")
ax.set_ylabel("Band gap (eV)")
ax.set_title(f"MP {prov['mp_db_version']}", fontsize=11)
fig.tight_layout(); fig.savefig("sol_3_6C_SiO_bandgaps.png", dpi=200)

# (2) Same OPTIMADE filter to three providers (no key needed for OPTIMADE)
providers = {
    "Materials Project": "https://optimade.materialsproject.org/v1",
    "OQMD": "https://oqmd.org/optimade/v1",
    "JARVIS": "https://jarvis.nist.gov/optimade/jarvisdft/v1",
}
flt = 'chemical_formula_reduced="O2Si"'
for name, base in providers.items():
    info = requests.get(base + "/info", timeout=30).json()
    r = requests.get(base + "/structures", params={"filter": flt, "page_limit": 1}, timeout=60)
    r.raise_for_status()
    meta = r.json()["meta"]
    prov[name] = {"api_version": meta.get("api_version"),
                  "data_returned": meta.get("data_returned"),
                  "data_available": meta.get("data_available"),
                  "provider_version": info.get("data", {}).get("attributes", {}).get("version")}
    print(f"{name:<18s} SiO2 entries = {meta.get('data_returned')}")

# (3) Box 3.6: write the provenance record next to the figure
json.dump(prov, open("sol_3_6C_provenance.json", "w"), indent=2, default=str)
print(json.dumps(prov, indent=2, default=str))
