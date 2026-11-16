# heat_pinn.jl — 1차원 확산 방정식 u_t = D u_xx, u(0, t) = u(1, t) = 0의 PINN (Lab 9.2, 9.3, 9.15 공용)
# 신경망 2-32-32-32-1 tanh. 점은 행벡터(1 x N)로 둔다: x, t가 1 x N이면 입력은 vcat(x, t) (2 x N).
#   soft: u = N(x, t),  손실 = PDE 잔차 + 초기조건 + 경계조건
#   hard: u = u0(x) + t x (1 - x) N(x, t)  (거리 함수 ansatz: 초기·경계조건이 정확히 성립, 손실은 잔차뿐)
include("ch9_common.jl")
include("pinn_common.jl")

const HEAT_SIZES = [2, 32, 32, 32, 1]
const EX, ET, Z2 = [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]          # x 방향, t 방향, 2차 입력 도함수(좌표 입력이므로 0)

"초기조건과 그 2계 도함수, 정확한 해"
struct HeatProblem{F1,F2,F3}
    u0::F1
    u0xx::F2
    exact::F3                                                   # (x, t, D) ↦ u
end
"Lab 9.2, 9.3: 두 모드의 합"
const HEAT2 = HeatProblem(x -> sin(pi * x) + 0.5 * sin(3pi * x), x -> -pi^2 * sin(pi * x) - 4.5 * pi^2 * sin(3pi * x),
                          (x, t, D) -> exp(-D * pi^2 * t) * sin(pi * x) + 0.5 * exp(-9 * D * pi^2 * t) * sin(3pi * x))
"Lab 9.15 (Ch.1 연습문제 1.8.D의 문제): 한 모드"
const HEAT1 = HeatProblem(x -> sin(pi * x), x -> -pi^2 * sin(pi * x), (x, t, D) -> exp(-D * pi^2 * t) * sin(pi * x))

"u, u_t, u_xx (각각 1 x N). 곱의 미분은 손으로 적었다: g = t x (1 - x), g_x = t (1 - 2x), g_xx = -2t, g_t = x (1 - x)"
function heat_derivs(pb, θ, x, t, hard)
    X = vcat(x, t)
    n, nx, nxx = jet2(HEAT_SIZES, θ, X, EX, Z2)
    _, nt = jet1(HEAT_SIZES, θ, X, ET)
    hard || return n, nt, nxx
    g = t .* x .* (1 .- x)
    u = pb.u0.(x) .+ g .* n
    ut = x .* (1 .- x) .* n .+ g .* nt
    uxx = pb.u0xx.(x) .- 2 .* t .* n .+ 2 .* t .* (1 .- 2 .* x) .* nx .+ g .* nxx
    return u, ut, uxx
end
function heat_residual(pb, θ, x, t, D, hard)
    _, ut, uxx = heat_derivs(pb, θ, x, t, hard)
    return ut .- D .* uxx
end
"값만 (평가용)"
function heat_u(pb, θ, x, t, hard)
    n = mlp_value(HEAT_SIZES, θ, vcat(x, t))
    return hard ? pb.u0.(x) .+ t .* x .* (1 .- x) .* n : n
end

"101 x 101 평가 격자 위의 상대 L2 오차, 최대 오차, 예측"
function heat_errors(pb, θ, hard; D = 0.1, n = 101)
    g = range(0, 1; length = n)
    xg = reshape([x for x in g, t in g], 1, :); tg = reshape([t for x in g, t in g], 1, :)
    up = heat_u(pb, θ, xg, tg, hard); ue = pb.exact.(xg, tg, D)
    return norm(up .- ue) / norm(ue), maximum(abs, up .- ue), reshape(up, n, n)
end

"배치점: 영역 안 nf개, 초기선 n0개, 경계 nb개(x = 0과 1에 절반씩)"
function heat_points(rng; nf = 1000, n0 = 200, nb = 200)
    xf, tf = rand(rng, 1, nf), rand(rng, 1, nf)
    x0 = rand(rng, 1, n0); tb = rand(rng, 1, nb)
    xb = hcat(zeros(1, nb ÷ 2), ones(1, nb - nb ÷ 2))
    return xf, tf, x0, xb, tb
end

"순방향 문제의 손실 θ ↦ L(θ)"
function heat_loss(pb, pts, hard; D = 0.1)
    xf, tf, x0, xb, tb = pts
    return function (θ)
        L = mean(abs2, heat_residual(pb, θ, xf, tf, D, hard))
        if !hard
            L += mean(abs2, mlp_value(HEAT_SIZES, θ, vcat(x0, zero(x0))) .- pb.u0.(x0))
            L += mean(abs2, mlp_value(HEAT_SIZES, θ, vcat(xb, tb)))
        end
        return L
    end
end

"Crank–Nicolson의 오차(자기 격자 위의 상대 L2, 최대 오차)와 가장 짧은 실행 시간"
function cn_error(pb, Nx, Nt; D = 0.1)
    x, U = crank_nicolson(D, Nx, Nt, pb.u0)
    E = [pb.exact(xi, tj, D) for xi in x, tj in range(0, 1; length = Nt + 1)]
    return norm(U .- E) / norm(E), maximum(abs, U .- E), maximum(abs, U[:, end] .- E[:, end]), best_time(() -> crank_nicolson(D, Nx, Nt, pb.u0); rep = 5)
end
