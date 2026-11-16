# lab_9_8.jl — Lab 9.8: FNO-1D로 Burgers 해 연산자 u0 ↦ u(·, T) 배우기, 해상도 전이의 정직한 시험
# (labs/93_data.py, 93_models.py, 93_fno_burgers.py)
# 줄인 포트: 솔버 검증과 자료 생성은 그대로, 학습은 s = 64 한 번(파이썬은 32/64/128 셋 + CNN + 절단·좌표 채널 절제).
# 자료: 512점 의사 스펙트럴 솔버(ν = 0.005, T = 0.25). FROM_PYTHON = true면 파이썬이 저장한 data/93_burgers.npz를 읽는다.
include("ch9_common.jl")
include("spectral1d.jl")
include("fno.jl")
using NPZ

const FROM_PYTHON = false
const NTR, NTE, NF, NU, T = 1000, 200, 512, 0.005, 0.25
const EPOCHS, BATCH = 50, 20

function validate_solver()
    for N in (64, 128)
        x = (0:N-1) ./ N
        u = burgers_solve(reshape(cole_hopf_exact.(x, 0.0, NU), N, 1), NU, 1.0, 5e-4)
        @printf("Cole-Hopf check  N=%4d dt=5e-4 T=1: max|u-u_exact| = %.2e\n", N, maximum(abs, vec(u) .- cole_hopf_exact.(x, 1.0, NU)))
    end
    u0 = grf(Xoshiro(0), 32, 512); u0h = rfft(u0, 1); u0h[end, :] .= 0     # 같은 대역 제한 초기조건을 두 격자에서
    u0 = irfft(u0h, 512, 1); u0f = irfft(vcat(u0h, zeros(ComplexF64, 256, 32)), 1024, 1) .* 2
    ref = burgers_solve(u0f, NU, 0.25, 1.25e-4)[1:2:end, :]
    for dt in (1e-3, 5e-4)
        e = rel_each(burgers_solve(u0, NU, 0.25, dt), ref)
        @printf("Burgers GRF N=512 dt=%.1e vs (N=1024, dt=1.25e-4): rel-L2 mean %.1e  max %.1e\n", dt, mean(e), maximum(e))
    end
end

"(u0, u(T)) 쌍: A, U는 (512, n)"
function load_data()
    if FROM_PYTHON
        traj = revdims(npzread(joinpath(CH9_DATA, "93_burgers.npz"))["traj"])   # (512, 11, 2200)
        return traj[:, 1, vcat(1:NTR, 2001:2000+NTE)], traj[:, end, vcat(1:NTR, 2001:2000+NTE)]
    end
    traj = burgers_dataset(; n = NTR + NTE, N = NF, nu = NU, T)
    @printf("Burgers: %d ICs, N=%d: mean drift max %.1e; u(T) rms fluctuation %.3f\n", NTR + NTE, NF,
            maximum(abs, mean(traj[:, end, :]; dims = 1) .- mean(traj[:, 1, :]; dims = 1)), mean(std(traj[:, end, :]; dims = 1, corrected = false)))
    return traj[:, 1, :], traj[:, end, :]
end
"점별 부분표본으로 s점 격자에, 채널 축을 붙여 (1, s, n)"
sub(X, s) = reshape(X[1:NF÷s:end, :], 1, s, :)

function main()
    validate_solver()
    A, U = load_data(); tr, te = 1:NTR, NTR+1:NTR+NTE
    RES = (32, 64, 128, 256, 512)
    @printf("data: %d train / %d test, nu=%g, T=%g; width 32, depth 4, %d epochs, batch %d\n", NTR, NTE, NU, T, EPOCHS, BATCH)
    s = 64; a, u = sub(A[:, tr], s), sub(U[:, tr], s)
    p0 = init_fno1d(Xoshiro(0); width = 32, kmax = 16, depth = 4)
    println("FNO parameters: ", count_params(p0))
    t0 = time()
    p, hist = fit_epochs(p0, (p, idx) -> rel_loss(fno1d(p, a[:, :, idx]), u[:, :, idx]), NTR; epochs = EPOCHS, batch = BATCH, lr = 3e-3, wd = 1e-5)
    test(r) = rel_each(fno1d(p, sub(A[:, te], r)), sub(U[:, te], r))
    @printf("FNO(kmax=16) trained at s=%3d (%4.0f s): test rel-L2 at s=", s, time() - t0)
    println(join([@sprintf("%d: %.4f", r, mean(test(r))) for r in RES], "  "))
    # 정직한 기준선: 학습 격자에서 예측한 뒤 Fourier 보간, 그리고 정확한 거친 해를 보간한 것(바닥)
    interp(X) = reshape(fourier_interp(dropdims(X; dims = 1), NF), 1, NF, :)
    u512 = sub(U[:, te], 512)
    @printf("  s=%3d -> 512: zero-shot %.4f | FNO@%d + Fourier interp %.4f | interp of exact coarse solution (floor) %.4f\n", s, mean(test(512)), s,
            mean(rel_each(interp(fno1d(p, sub(A[:, te], s))), u512)), mean(rel_each(interp(sub(U[:, te], s)), u512)))
    # 구조 점검: 이산화 자기 일관성(에일리어싱), 평행이동 등변성, 평균 보존
    lo = fno1d(p, sub(A[:, te], 64))
    for r in (128, 512)
        @printf("self-consistency: ||G(a_%d)|_64 - G(a_64)|| / ||G(a_64)|| = %.4f\n", r, mean(rel_each(fno1d(p, sub(A[:, te], r))[:, 1:r÷64:end, :], lo)))
    end
    a64 = sub(A[:, te], 64); sh = 5
    eq = mean(rel_each(fno1d(p, circshift(a64, (0, sh, 0))), circshift(fno1d(p, a64), (0, sh, 0))))
    dm = abs.(vec(mean(fno1d(p, a64); dims = 2)) .- vec(mean(a64; dims = 2)))
    @printf("translation equivariance error (shift %d/64): %.1e\n", sh, eq)
    @printf("mean conservation |mean(pred) - mean(u0)|: mean %.2e, max %.2e  (solver: 1e-15)\n", mean(dm), maximum(dm))
    # 비용: FNO 순전파 대 솔버(같은 정확도의 거친 솔버까지)
    t_inf = best_time(() -> fno1d(p, a64); rep = 20) / NTE
    At = Float64.(A[:, te]); Ut = Float64.(U[:, te])
    t_sol = @elapsed(burgers_solve(At, NU, T, 5e-4)) / NTE
    @printf("cost per sample: FNO forward (s=64) %.3f ms | reference solver N=512, dt=5e-4: %.1f ms\n", 1e3 * t_inf, 1e3 * t_sol)
    for (Nc, dtc) in ((64, 2.5e-3), (96, 1.25e-3), (128, 1.25e-3))
        ac = NF % Nc == 0 ? At[1:NF÷Nc:end, :] : fourier_interp(At, Nc)
        t0 = time(); uc = burgers_solve(ac, NU, T, dtc); tc = time() - t0
        @printf("   coarse solver N=%3d, dt=%.2e: %.2f ms, rel-L2 at 512 (after Fourier interpolation) %.4f\n", Nc, dtc, 1e3 * tc / NTE, mean(rel_each(fourier_interp(uc, NF), Ut)))
    end
    serialize(joinpath(CH9_OUT, "93_fno_burgers_s64.jls"), p)
end
main()
