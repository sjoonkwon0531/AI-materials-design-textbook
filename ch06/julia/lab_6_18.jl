# Lab 6.18 (Julia) — 같은 조성, 다른 배열: 조성 기술자 대 SOAP
# 파이썬 labs/67_soap_emt.py 대응(짧은 포트). Cu-Au fcc 32원자 세포의 생성 규칙(무작위·층상 규칙·편석,
# 반자리 교환, ±2% 변형, 0.03 Å 흔들기), 쌍 특성의 이웃 목록, SOAP 풀링(원자 평균 'outer', 중심 원소별 합),
# 표준화 + RidgeCV(효율적 LOO로 alpha 선택), 5겹 교차검증, 학습 곡선은 Julia로 옮겼다.
# EMT 에너지(ASE)와 원자별 SOAP·'inner' 평균 SOAP(DScribe)는 Julia 대응이 없어 PythonCall로 부른다.
# 파이썬 판의 RBF-KRR 행(그리드 탐색)은 결론을 바꾸지 않으므로 생략했다.
# 구조가 Julia RNG로 만들어지므로 숫자는 통계적으로 일치한다. 볼 것: 조성 ~8 meV/atom, 손으로 만든 쌍 특성
# ~3, inner ~6, outer ~0.3, 중심 원소별 합 ~0.1 meV/atom의 순서와 자릿수.
using LinearAlgebra, Printf, Random, Statistics
include("ch6_common.jl")
include("descriptors.jl")
include("dscribe_bridge.jl")
load_dscribe_shim(LABS)

const KINDS = (:random, :ordered, :segregated)

function make(rng, nAu, kind, strain)
    a = 3.61 + (4.08 - 3.61) * nAu / 32                                   # Vegard 격자상수
    s = fcc_supercell(29, a, 2); z = (s.cell \ s.R)[3, :]
    order = kind == :random ? randperm(rng, 32) :
            kind == :ordered ? sortperm(collect(zip(mod.(round.(Int, z .* 4), 2), rand(rng, 32)))) :
            sortperm(collect(zip(z, rand(rng, 32))))                     # np.lexsort((r, key))와 같은 순서
    sym = fill(29, 32); sym[order[1:nAu]] .= 79
    for _ in 1:rand(rng, 0:4)
        if 0 < nAu < 32
            i = rand(rng, findall(==(29), sym)); j = rand(rng, findall(==(79), sym))
            sym[i], sym[j] = 79, 29
        end
    end
    f = strain ? 1 + 0.04 * rand(rng) - 0.02 : 1.0
    return Structure(sym, f .* s.R .+ 0.03 .* randn(rng, 3, 32), f .* s.cell)
end

"표준화 + ridge, alpha는 학습 자료의 효율적 LOO(SVD)로 고른다 (sklearn RidgeCV의 기본 동작과 같은 기준)"
function ridgecv_predict(Xtr, ytr, Xte; alphas = 10 .^ range(-6, 3; length = 19))
    sc = scaler(Xtr); Z = zscore_with(Xtr, sc); mz = vec(mean(Z; dims = 1)); my = mean(ytr)
    Zc = Z .- mz'; yc = ytr .- my; n = length(ytr)
    F = svd(Zc); Uy = F.U' * yc
    best = (Inf, 0.0)
    for a in alphas
        f = F.S .^ 2 ./ (F.S .^ 2 .+ a)
        h = 1 / n .+ vec(sum((F.U .^ 2) .* f'; dims = 2))                # 절편 포함 hat 행렬의 대각
        r = yc .- F.U * (f .* Uy)
        loo = mean((r ./ (1 .- h)) .^ 2)
        loo < best[1] && (best = (loo, a))
    end
    w = ridge_fit(Z, ytr, best[2])
    return ridge_predict(w, zscore_with(Xte, sc))
end

function pair_fractions(s; rc = 3.2)
    diff = 0; auau = 0; tot = 0
    for i in 1:natoms(s), (j, _) in neighbor_vectors(s, i, rc)
        tot += 1; diff += s.Z[i] != s.Z[j]; auau += (s.Z[i] == 79) & (s.Z[j] == 79)
    end
    return [diff / tot, auau / tot]
end

oof(F, y; folds = kfold(length(y), 5; rng = MersenneTwister(0))) = cv_predict(ridgecv_predict, F, y, folds)

function main()
    t0 = time()
    soap = dscribe_desc.SOAP(; species = pylist(["Cu", "Au"]), r_cut = 5.0, n_max = 6, l_max = 4, sigma = 0.5, periodic = true)
    soap_in = dscribe_desc.SOAP(; species = pylist(["Cu", "Au"]), r_cut = 5.0, n_max = 6, l_max = 4, sigma = 0.5,
                                periodic = true, average = "inner")
    saved = nothing
    for strain in (false, true)
        rng = MersenneTwister(0); structs = Structure[]; x = Float64[]; kinds = Symbol[]
        for k in 0:399
            nAu = rand(rng, 0:32); kind = KINDS[k%3+1]
            push!(structs, make(rng, nAu, kind, strain)); push!(x, nAu / 32); push!(kinds, kind)
        end
        y = 1000 .* emt_energy_per_atom.(structs)
        @printf("=== strain %s: N=%d cells; E/atom std %.1f meV\n", strain ? "+-2%" : "off", length(y), std(y; corrected = false))
        if !strain
            for kd in (:ordered, :random, :segregated)
                m = (kinds .== kd) .& (x .> 0.3) .& (x .< 0.7)
                @printf("   %-10s cells with 0.3<x<0.7: n=%d, mean E/atom %.1f meV\n", kd, sum(m), mean(y[m]))
            end
        end
        atoms = pylist([to_ase(s) for s in structs])
        P = pyconvert(Array{Float64,3}, soap.create(atoms; n_jobs = 1))          # (N, 32, 390)
        S_inn = pyconvert(Matrix{Float64}, soap_in.create(atoms; n_jobs = 1))
        Zs = permutedims(reduce(hcat, [s.Z for s in structs]))                    # N x 32
        S_avg = dropdims(mean(P; dims = 2); dims = 2)
        S_res = hcat(dropdims(sum(P .* (Zs .== 29); dims = 2); dims = 2), dropdims(sum(P .* (Zs .== 79); dims = 2); dims = 2)) ./ 32
        pair = permutedims(reduce(hcat, pair_fractions.(structs)))
        vol = [abs(det(s.cell)) / 32 for s in structs]
        X1 = hcat(x, x .^ 2, x .^ 3); X2 = hcat(X1, pair)
        strain && (X1 = hcat(X1, vol); X2 = hcat(X2, vol))
        @printf("   SOAP per atom: %d dims\n", size(P, 3))
        @printf("   baseline (predict mean)                      MAE %6.2f meV/atom\n", mae(y, fill(mean(y), length(y))))
        for (lab, F) in (("composition$(strain ? " + volume" : ""), cubic ridge", X1), ("+ NN pair fractions (hand-built), ridge", X2),
                         ("SOAP, DScribe average='inner', ridge", S_inn), ("SOAP, power spectra averaged ('outer'), ridge", S_avg),
                         ("SOAP summed per centre species, ridge", S_res))
            @printf("   %-44s MAE %6.2f meV/atom\n", lab, mae(oof(F, y), y))
        end
        strain || (saved = (y, X1, S_res))
    end
    y, X1, S_res = saved
    println("learning curve (strain off; 80 held-out cells; 5 random draws):")
    rng = MersenneTwister(1)
    for n in (10, 20, 40, 80, 160, 320)
        r1, r2 = Float64[], Float64[]
        for _ in 1:5
            idx = randperm(rng, 400); te, tr = idx[1:80], idx[81:80+n]
            push!(r1, mae(ridgecv_predict(X1[tr, :], y[tr], X1[te, :]), y[te]))
            push!(r2, mae(ridgecv_predict(S_res[tr, :], y[tr], S_res[te, :]), y[te]))
        end
        @printf("   n_train=%3d  composition %.2f+/-%.2f   SOAP(centre-resolved) %.2f+/-%.2f meV/atom\n",
                n, mean(r1), std(r1; corrected = false), mean(r2), std(r2; corrected = false))
    end
    @printf("wall %.0f s\n", time() - t0)
end

main()
