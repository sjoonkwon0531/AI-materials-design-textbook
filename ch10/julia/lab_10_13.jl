# lab_10_13.jl: Lab 10.13: 변위 크기에 대한 수렴, 그리고 같은 곡률을 두 길로 구하기 (labs/107_check_disp.py, 107_compare.py의 끝)
# 실행: julia lab_10_13.jl mace | chgnet | sevennet | mattersim   (A) a = 3.944 Å에서 변위 0.005–0.04 Å의 최저 진동수
#       julia lab_10_13.jl fit                                     (B) 강체 회전 곡선 E = c₂u² + c₄u⁴의 곡률 대 유한 변위 R4+
# (B)는 lab_10_12.jl이 만든 models/107_<model>.json을 읽는다(FROM_PYTHON=true이면 파이썬 판의 data/107_<model>.json).
# 힘 상수는 대칭을 쓰지 않는 ±d 변위 30개로 만든다. phonopy는 대칭으로 고른 변위 3개(산소는 (1,0,1)/√2 방향)를 쓰므로,
# 비조화 항이 섞이는 큰 변위(0.04 Å)에서는 두 방법의 값이 몇 cm⁻¹ 달라진다.
const ARG = isempty(ARGS) ? "mace" : ARGS[1]
ENV["CH10_MODEL"] = ARG == "fit" ? "mace" : ARG
include("ch10_common2.jl")

function scan(model)
    dtype = model == "mace" ? "float64" : "float32"
    calc, _ = make_calc(model; dtype); c = Counted(calc)
    efun(s) = energy_forces(c, s)
    @printf("=== %s (%s), a = 3.944 A: lowest frequency (cm-1) at G, X, M, R ===\n", LABEL[model], dtype)
    res = Dict{String,Any}()
    for d in (0.005, 0.01, 0.02, 0.04)
        fr, _ = phonons(efun, sto(3.944); rep = 2, d)
        low = Dict(k => fr[k][1] for k in QORDER)
        acc = maximum(sort(abs.(fr["G"]))[1:3])                  # Γ의 음향 모드 세 개는 0이어야 한다
        res[string(d)] = low
        println(@sprintf("  disp %5.3f A: ", d), join([@sprintf("%s %5.1f%s", k, abs(low[k]), low[k] < 0 ? "i" : " ") for k in QORDER], "  "),
                @sprintf("   |acoustic at G| max %.2f", acc))
    end
    writejson(joinpath(JOUT, "107_disp_$(model).json"), res)
end

"""(B) 회전각 6° 이하에서 화학식 단위당 E = c₂u² + c₄u⁴, u = (a/2) tan φ (a는 모델 자신의 a₀).
화학식 단위마다 적도 산소 두 개가 u만큼 움직이므로 운동 에너지는 m_O u̇², 따라서 ω² = c₂ / m_O."""
function curvature_fit(; from_python = false)
    println("=== two routes to the R4+ curvature: frozen tilt profile E(u) versus finite-displacement forces ===")
    println("  fit E = c2 u^2 + c4 u^4 per f.u. for tilt <= 6 deg, omega^2 = c2 / m_O;  1 sqrt(eV/(A^2 amu)) = 521.47 cm-1")
    for m in MODELS
        d = readjson(joinpath(from_python ? DATA : JOUT, "107_$(m).json"))
        t = d["tilt"]; a = d["eos"]["a0"]
        phi = Float64.(t["phis"]); E = Float64.(t["prof"]) ./ 1000
        u = a / 2 .* tand.(phi); sel = phi .<= 6
        c2, c4 = [u[sel] .^ 2 u[sel] .^ 4] \ E[sel]               # 원점을 지나는 최소제곱 (상수항 없음)
        w = sign(c2) * sqrt(abs(c2) / MASS[8]) * SQRT_EV_A2_AMU_CM
        wR = minimum(Float64.(d["ph"]["own"]["freqs"]["R"]))
        @printf("  %-16s c2 = %+.3f eV/A^2, c4 = %.2f eV/A^4 -> %4s cm-1 ; finite-displacement R4+ (0.01 A) %4s cm-1\n",
                LABEL[m], c2, c4, fmt_w(w), fmt_w(wR))
    end
end

ARG == "fit" ? curvature_fit(; from_python = get(ENV, "FROM_PYTHON", "false") == "true") : scan(ARG)
