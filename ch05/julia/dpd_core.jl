# dpd_core.jl — 최소 3D DPD 엔진 (Groot & Warren 1997 규약), 셀 리스트 + 반껍질. 파이썬 labs/dpd_core.py 대응.
# 단위 r_c = m = k_BT = 1.  F^C = a_ij (1-r) r̂,  F^D = -gamma w^2 (r̂.v_ij) r̂,  F^R = sigma w theta_ij/sqrt(dt) r̂,
# w = 1 - r (r < 1), sigma^2 = 2 gamma k_BT.  결합: F = -C r_ij (조화, 평형 길이 0).
# 적분기: GW 수정 velocity Verlet(매개변수 lambda). 외부장(화학 패턴)은 run!의 키워드 h, kx, mode, w.
# 배열: x, v, f는 3 x n (열 = 비드), typ는 0(A)/1(B), bonds는 2 x nb (1-based 비드 번호).
# 셀·이웃 계산은 파이썬과 같은 0-based 정수로 하고 배열 접근에서만 +1 한다. 연결 리스트의 끝은 0.
using Random, Statistics

const HS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (-1, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
            (-1, 1, 1), (-1, 0, 1), (-1, -1, 1), (0, -1, 1), (1, -1, 1))   # 자기 셀 + 앞쪽 이웃 13개

"힘 f와 보존력+결합의 virial 대각 성분 ptens를 채운다. 쌍 난수 theta_ij는 쌍을 만날 때 한 번 뽑는다."
function forces!(f, ptens, x, v, typ, amat, gamma, sigma, dt, box, bonds, Cb)
    n = size(x, 2)
    nc = (trunc(Int, box[1]), trunc(Int, box[2]), trunc(Int, box[3]))   # 각 방향 3개 이상이어야 한다
    cl = (box[1] / nc[1], box[2] / nc[2], box[3] / nc[3])
    ncell = nc[1] * nc[2] * nc[3]
    head = zeros(Int, ncell); nxt = zeros(Int, n)
    @inbounds for i in 1:n
        c0 = mod(trunc(Int, x[1, i] / cl[1]), nc[1]); c1 = mod(trunc(Int, x[2, i] / cl[2]), nc[2])
        c2 = mod(trunc(Int, x[3, i] / cl[3]), nc[3])
        c = (c0 * nc[2] + c1) * nc[3] + c2 + 1
        nxt[i] = head[c]; head[c] = i
    end
    fill!(f, 0.0); fill!(ptens, 0.0)
    isq = 1.0 / sqrt(dt)
    @inbounds for c in 0:ncell-1
        c2 = c % nc[3]; c1 = (c ÷ nc[3]) % nc[2]; c0 = c ÷ (nc[2] * nc[3])
        for o in 1:14
            e0 = mod(c0 + HS[o][1], nc[1]); e1 = mod(c1 + HS[o][2], nc[2]); e2 = mod(c2 + HS[o][3], nc[3])
            cc = (e0 * nc[2] + e1) * nc[3] + e2
            i = head[c+1]
            while i > 0
                j = o == 1 ? nxt[i] : head[cc+1]
                while j > 0
                    dx = x[1, i] - x[1, j]; dy = x[2, i] - x[2, j]; dz = x[3, i] - x[3, j]
                    dx -= box[1] * round(dx / box[1]); dy -= box[2] * round(dy / box[2]); dz -= box[3] * round(dz / box[3])
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 < 1.0
                        r = sqrt(r2); ex = dx / r; ey = dy / r; ez = dz / r; w = 1.0 - r
                        vr = ex * (v[1, i] - v[1, j]) + ey * (v[2, i] - v[2, j]) + ez * (v[3, i] - v[3, j])
                        fc = amat[typ[i]+1, typ[j]+1] * w
                        fm = fc - gamma * w * w * vr + sigma * w * randn() * isq
                        f[1, i] += fm * ex; f[2, i] += fm * ey; f[3, i] += fm * ez
                        f[1, j] -= fm * ex; f[2, j] -= fm * ey; f[3, j] -= fm * ez
                        ptens[1] += fc * ex * dx; ptens[2] += fc * ey * dy; ptens[3] += fc * ez * dz
                    end
                    j = nxt[j]
                end
                i = nxt[i]
            end
        end
    end
    @inbounds for b in 1:size(bonds, 2)
        i = bonds[1, b]; j = bonds[2, b]
        dx = x[1, i] - x[1, j]; dy = x[2, i] - x[2, j]; dz = x[3, i] - x[3, j]
        dx -= box[1] * round(dx / box[1]); dy -= box[2] * round(dy / box[2]); dz -= box[3] * round(dz / box[3])
        f[1, i] -= Cb * dx; f[2, i] -= Cb * dy; f[3, i] -= Cb * dz
        f[1, j] += Cb * dx; f[2, j] += Cb * dy; f[3, j] += Cb * dz
        ptens[1] -= Cb * dx * dx; ptens[2] -= Cb * dy * dy; ptens[3] -= Cb * dz * dz
    end
    return f
end

"""GW 수정 velocity Verlet. every step마다 [kT, Pxx, Pyy, Pzz] 한 행 (kT = sum v^2/(3(n-1)), P는 운동 부분 포함).
외부장 U = -h e_j phi(x_j), e_j = +1(A), -1(B). mode 0: phi = cos(kx x); mode 1: 폭 w의 Gauss 줄무늬, 간격 2pi/kx.
h = 0이면 파이썬 run()과 같고, h != 0이면 run_field()와 같다 (외부장은 virial에서 제외)."""
function run!(x, v, f, typ, amat, gamma, sigma, dt, lam, box, bonds, Cb, nstep, every, ptens;
              h = 0.0, kx = 1.0, mode = 0, w = 1.0)
    n = size(x, 2); V = box[1] * box[2] * box[3]
    out = zeros(nstep ÷ every, 4); vt = similar(v); fo = similar(f)
    for s in 1:nstep
        @inbounds for i in 1:n, d in 1:3
            x[d, i] += dt * v[d, i] + 0.5 * dt * dt * f[d, i]
            x[d, i] -= box[d] * floor(x[d, i] / box[d])
            vt[d, i] = v[d, i] + lam * dt * f[d, i]          # 예측 속도 (lambda)
            fo[d, i] = f[d, i]
        end
        forces!(f, ptens, x, vt, typ, amat, gamma, sigma, dt, box, bonds, Cb)
        if h != 0.0
            @inbounds for i in 1:n
                e = typ[i] == 0 ? 1.0 : -1.0
                if mode == 0
                    f[1, i] -= h * e * kx * sin(kx * x[1, i])
                else
                    P = 2pi / kx; d = x[1, i] - P * round(x[1, i] / P)
                    f[1, i] -= h * e * d / (w * w) * exp(-0.5 * d * d / (w * w))
                end
            end
        end
        @inbounds for i in 1:n, d in 1:3
            v[d, i] += 0.5 * dt * (fo[d, i] + f[d, i])
        end
        if s % every == 0
            k = s ÷ every
            ke = vec(sum(v .* v; dims = 2))
            out[k, 1] = sum(ke) / (3.0 * (n - 1))
            for d in 1:3
                out[k, 1+d] = (ke[d] + ptens[d]) / V
            end
        end
    end
    return out
end

function init_random(n, box)
    x = rand(3, n) .* box
    v = randn(3, n); v .-= mean(v; dims = 2)
    return x, v
end

make_amat(a_like, da) = [a_like a_like+da; a_like+da a_like]

const NOB = zeros(Int, 2, 0)                                  # 결합 없음
