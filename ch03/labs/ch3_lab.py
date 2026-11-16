"""ch3_lab.py — Chapter 3 §3.6 Hands-on Lab 전체 코드 (Lab 3.1–3.3, OPTIMADE).
Lab 3.1·3.3은 로컬 계산(pymatgen)만 쓴다. Lab 3.2는 Materials Project API 키(MP_API_KEY 환경변수)가,
OPTIMADE 부분은 인터넷 접속이 필요하다. 각 PART는 본문의 코드 블록을 등장 순서대로 옮긴 것이다."""

# ======================================================================
# PART A — Lab 3.1 — pymatgen으로 구조·조성 다루기 (로컬, 실행 검증)
# ======================================================================
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

a = 5.43  # Si 격자상수 (Angstrom)
si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(a), ["Si"], [[0, 0, 0]])
sga = SpacegroupAnalyzer(si)
print(f"reduced formula : {si.composition.reduced_formula}")
print(f"n_sites         : {len(si)}")
print(f"density (g/cm^3): {si.density:.3f}")
print(f"space group     : {sga.get_space_group_symbol()} {sga.get_space_group_number()}")
print(f"lattice a (A)   : {si.lattice.a:.2f} | volume (A^3): {si.volume:.1f}")

# ======================================================================
# PART B — Lab 3.1 — pymatgen으로 구조·조성 다루기 (로컬, 실행 검증)
# ======================================================================
c = Composition("Fe2O3")
print(f"reduced formula : {c.reduced_formula}")
print(f"molar weight    : {c.weight:.2f} g/mol")
print(f"atomic frac Fe  : {c.get_atomic_fraction('Fe'):.3f}")

# ======================================================================
# PART C — Lab 3.2 — Materials Project API로 물성 조회 (코드 제시; 키 필요)
# ======================================================================
from mp_api.client import MPRester   # pip install mp-api

with MPRester() as mpr:               # 키는 MP_API_KEY 환경변수에서 읽힘
    doc = mpr.materials.summary.search(
        material_ids=["mp-149"],      # mp-149 = 실리콘
        fields=["formula_pretty", "band_gap"],
    )[0]
    print(doc.formula_pretty, doc.band_gap)

# ======================================================================
# PART D — Lab 3.2 — Materials Project API로 물성 조회 (코드 제시; 키 필요)
# ======================================================================
with MPRester() as mpr:
    docs = mpr.materials.summary.search(
        chemsys="Si-O",
        is_stable=True,          # 서버 측에서 hull 위(E_hull=0) 상만
        include_gnome=False,     # 기본값 True(포함); False면 GNoME 유래 항목 제외
        fields=["material_id", "formula_pretty", "band_gap", "energy_above_hull"],
    )

# ======================================================================
# PART E — Lab 3.3 — convex hull로 상안정성 판정 (로컬, 실행 검증)
# ======================================================================
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

entries = [                                  # 원소는 0 기준 => 형성에너지 hull
    PDEntry(Composition("Li"),    0.0),
    PDEntry(Composition("O"),     0.0),
    PDEntry(Composition("Li2O"),  -2.05 * 3),   # ~ -2.05 eV/atom
    PDEntry(Composition("Li2O2"), -1.58 * 4),   # ~ -1.58 eV/atom
    PDEntry(Composition("LiO2"),  -0.90 * 3),   # ~ -0.90 eV/atom
]
pd = PhaseDiagram(entries)
for e in entries:
    x_O = e.composition.get_atomic_fraction("O")
    ef, eh = pd.get_form_energy_per_atom(e), pd.get_e_above_hull(e)
    tag = "STABLE" if eh < 1e-8 else f"{eh:+.3f} eV/atom above hull"
    print(f"  {e.composition.reduced_formula:<6s} x_O={x_O:.3f} E_form={ef:+.3f} eV/atom -> {tag}")

# ======================================================================
# PART F — OPTIMADE — 여러 DB를 한 질의로 (코드 제시; 네트워크 필요)
# ======================================================================
import requests
providers = {
    "Materials Project": "https://optimade.materialsproject.org/v1/structures",
    "OQMD":              "https://oqmd.org/optimade/v1/structures",
    "JARVIS":            "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures",
}
flt = 'chemical_formula_reduced="O2Si"'
for name, base in providers.items():
    r = requests.get(base, params={"filter": flt, "page_limit": 5}, timeout=30)
    r.raise_for_status()
    print(name, r.json()["meta"]["data_returned"])

# ======================================================================
# PART G — OPTIMADE — 여러 DB를 한 질의로 (코드 제시; 네트워크 필요)
# ======================================================================
with MPRester() as mpr:
    print(mpr.db_version)   # DB 버전 문자열 (get_database_version()은 폐기 예정)
