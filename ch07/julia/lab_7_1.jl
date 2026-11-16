# lab_7_1.jl — Lab 7.1: ReLU 퍼텐셜은 에너지를 맞혀도 힘과 에너지 보존을 망친다 (labs/71_activation.py)
# Morse O–H 곡선을 에너지 + 힘으로 맞추고(힘 = -dE/dr를 손실에 넣으므로 2계 미분이 필요: Lux 중첩 AD),
# 힘의 불연속, 평형 힘상수 k = d²E/dr², 1차원 NVE velocity Verlet의 에너지 보존 차수를 본다.
include("ch7_common.jl")
using Lux, Zygote, ForwardDiff, SpecialFunctions
import Optimisers

const De, aM, re = 4.0, 2.0, 1.0                      # eV, 1/Å, Å
const mu = 1.008 * 15.999 / (1.008 + 15.999)          # 환산 질량, amu
const CONV = 0.0096485                                # (eV/amu) -> Å²/fs²
morse(r) = De * (1 - exp(-aM * (r - re)))^2
morse_f(r) = (e = exp(-aM * (r - re)); -2De * aM * e * (1 - e))   # F = -dE/dr

const r_tr = collect(range(0.80, 1.60; length = 41))
const E_tr, F_tr = morse.(r_tr), morse_f.(r_tr)
"PyTorch nn.GELU 기본(erf 정확식). NNlib의 gelu는 tanh 근사이므로 따로 정의했다"
gelu_exact(x) = x * (1 + erf(x / sqrt(oftype(x, 2)))) / 2
const ACTS = ["ReLU" => relu, "GELU" => gelu_exact, "SiLU" => swish, "tanh" => tanh]   # swish = SiLU

"PyTorch식 균일 초기화(fan_in)를 쓰는 Float64 Dense"
tu(nin) = (rng, dims...) -> (rand(rng, Float64, dims...) .* 2 .- 1) ./ sqrt(nin)
ldense(nin, nout, act = identity) = Dense(nin => nout, act; init_weight = tu(nin), init_bias = tu(nin))

"입력 척도 (r - 1.2)/0.25를 첫 층에 넣어 두면 내부 기울기가 곧바로 dE/dr가 된다"
make(act) = Chain(WrappedFunction(r -> (r .- 1.2) ./ 0.25), ldense(1, 64, act), ldense(64, 64, act), ldense(64, 1))

"E (1 x N)와 F = -dE/dr. 손실 안에서 불리면 Lux가 안쪽 Zygote.gradient를 forward-over-reverse로 바꾼다"
function EF(model, ps, st, r::AbstractMatrix)
    sm = StatefulLuxLayer{true}(model, ps, st)
    E = sm(r)
    F = -only(Zygote.gradient(sum ∘ sm, r))
    return vec(E), vec(F)
end

function train(model, ps, st; steps = 4000, lr = 2e-3)
    x = reshape(r_tr, 1, :)
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    function loss(p)
        E, F = EF(model, p, st, x)
        return mean(abs2, E .- E_tr) + 0.1 * mean(abs2, F .- F_tr)
    end
    for s in 1:steps
        Optimisers.adjust!(opt, cosine_lr(lr, s - 1, steps))
        g = only(Zygote.gradient(loss, ps))
        opt, ps = Optimisers.update!(opt, ps, g)
    end
    return ps
end

"스칼라 함수 E(r)와 그 도함수 (평가는 ForwardDiff로; 2계 도함수 k도 Dual 중첩으로 정확하다)"
energy_fn(model, ps, st) = r -> first(first(model(fill(r, 1, 1), ps, st)))
force_fn(Ef) = r -> -ForwardDiff.derivative(Ef, r)
kconst_fn(Ef) = r -> ForwardDiff.derivative(q -> ForwardDiff.derivative(Ef, q), r)

"1차원 NVE velocity Verlet: max |E_tot(t) - E_tot(0)| (eV)"
function verlet(Ef, r0, dt; T = 2000.0)
    Ff = force_fn(Ef)
    r, v = r0, 0.0
    F = Ff(r); E0 = Ef(r); dmax = 0.0
    for _ in 1:round(Int, T / dt)
        v += 0.5 * dt * F / mu * CONV; r += dt * v
        F = Ff(r); v += 0.5 * dt * F / mu * CONV
        dmax = max(dmax, abs(Ef(r) + 0.5 * mu * v^2 / CONV - E0))
    end
    return dmax
end

function main()
    rg = collect(range(0.80, 1.60; length = 8001))      # 1e-4 Å 간격
    @printf("Morse De=%.1f eV, a=%.1f 1/A, re=%.1f A, mu=%.4f amu; true k = %.2f eV/A^2\n", De, aM, re, mu, 2De * aM^2)
    @printf("%-5s %4s %11s %12s %18s %8s\n", "act", "seed", "RMSE_E(meV)", "RMSE_F(eV/A)", "max|dF| per 1e-4A", "k at re")
    dts = [0.4, 0.2, 0.1, 0.05]; drift = Dict{String,Vector{Float64}}()
    for (name, act) in ACTS, seed in 0:2
        model = make(act)
        ps, st = Lux.setup(Xoshiro(seed), model)
        ps = train(model, ps, st)
        Ef = energy_fn(model, ps, st)
        E, F = EF(model, ps, st, reshape(rg, 1, :))
        ire = argmin(abs.(rg .- re))
        @printf("%-5s %4d %11.2f %12.4f %18.4f %8.2f\n", name, seed, rmse(E, morse.(rg)) * 1e3, rmse(F, morse_f.(rg)),
                maximum(abs.(diff(F))), kconst_fn(Ef)(rg[ire]))
        seed == 0 && (drift[name] = [verlet(Ef, 1.15, dt) for dt in dts])
    end
    println("NVE velocity Verlet, r0 = 1.15 A at rest, 2000 fs: max |E_tot - E_tot(0)| (meV) vs dt (fs) [seed 0]")
    for (name, _) in ACTS
        d = drift[name] .* 1e3
        A = [log.(dts) ones(4)]; slope = (A \ log.(d))[1]
        println(rpad(name, 6), join([@sprintf("%10.4f", x) for x in d]), @sprintf("   log-log slope %5.2f", slope))
    end
end

main()
