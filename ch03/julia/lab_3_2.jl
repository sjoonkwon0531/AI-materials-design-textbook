# Lab 3.2 (Julia) — Materials Project API로 물성 조회 (키 필요; 네트워크 필요)
# 공식 클라이언트 mp-api는 파이썬 패키지이므로, PythonCall.jl로 그대로 호출하는 것이
# 가장 정직한 이식이다(경로 A). 파이썬 없이 쓰고 싶다면 MP의 REST 엔드포인트를
# HTTP.jl + JSON3.jl로 직접 부를 수도 있다(경로 B). 두 경로 모두 환경변수 MP_API_KEY를 쓴다.
#   준비:  ] add PythonCall HTTP JSON3   그리고  CondaPkg로 mp-api 설치:
#          using CondaPkg; CondaPkg.add_pip("mp-api")

# ============ 경로 A: PythonCall.jl + mp-api (파이썬 판과 1:1) ============
using PythonCall
mpc = pyimport("mp_api.client")

pywith(mpc.MPRester()) do mpr                 # 파이썬의 `with MPRester() as mpr:`
    docs = mpr.materials.summary.search(
        material_ids = pylist(["mp-149"]),    # mp-149 = 실리콘
        fields = pylist(["formula_pretty", "band_gap"]),
    )
    doc = docs[0]                             # 주의: Py 객체 인덱싱은 파이썬 규칙(0-based)
    println(pyconvert(String, doc.formula_pretty), " ", pyconvert(Float64, doc.band_gap))
    println("database version: ", pyconvert(String, pystr(mpr.db_version)))   # Box 3.6: 버전을 기록하라

    # 화학계 단위 질의: Si-O계 안정상만
    docs2 = mpr.materials.summary.search(
        chemsys = "Si-O",
        is_stable = true,                     # 서버 측에서 hull 위(E_hull=0) 상만
        include_gnome = false,                # 기본값 True; false면 GNoME 유래 항목 제외
        fields = pylist(["material_id", "formula_pretty", "band_gap", "energy_above_hull"]),
    )
    for d in docs2                            # Py 리스트는 Julia for 루프로 그대로 순회된다
        println(pystr(d.material_id), "  ", pyconvert(String, d.formula_pretty),
                "  Eg=", pyconvert(Float64, d.band_gap), " eV  E_hull=",
                pyconvert(Float64, d.energy_above_hull))
    end
end

# ============ 경로 B: 순수 Julia (HTTP.jl + JSON3.jl로 REST 직접 호출) ============
# using HTTP, JSON3
# key = ENV["MP_API_KEY"]
# r = HTTP.get("https://api.materialsproject.org/materials/summary/";
#              query = ["material_ids" => "mp-149", "_fields" => "formula_pretty,band_gap"],
#              headers = ["X-API-KEY" => key], readtimeout = 30)
# d = JSON3.read(r.body).data[1]              # Julia 배열이므로 1-based
# println(d.formula_pretty, " ", d.band_gap)
