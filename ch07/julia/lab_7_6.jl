# lab_7_6.jl — Lab 7.6: 2D CNN은 미세조직을 옮겨도 알아보지만 돌리면 못 알아본다 (labs/72a_micro_cnn.py)
# 띠통과 걸러진 잡음을 문턱으로 자른 64x64 '미세조직' 세 종류(라멜라, 미로, 액적). FFTW.jl로 거르고,
# scipy.ndimage.rotate(mode = "wrap", order = 1)는 주기 경계 쌍선형 보간 회전으로 직접 썼다.
# 배치: Flux 2D 합성곱 입력은 (64, 64, 채널, 배치). circular padding은 NNlib.pad_circular.
include("ch7_common.jl")
include("flux_tools.jl")
using FFTW

const L = 64
const KX = fftfreq(L) .* ones(L)'; const KY = ones(L) .* fftfreq(L)'
const KK = sqrt.(KX .^ 2 .+ KY .^ 2); const PHI = atan.(KY, KX)

"cls 0: 라멜라(방향 theta, 도), 1: 등방 미로(면적 50%), 2: 액적(~20%)"
function micro(cls, rng; theta = 0.0)
    k0 = 0.09 + 0.05 * rand(rng)
    ring = exp.(-0.5 .* ((KK .- k0) ./ 0.02) .^ 2)
    filt = cls == 0 ? ring .* exp.(-0.5 .* (angle.(exp.(2im .* (PHI .- deg2rad(theta)))) ./ (2 * deg2rad(12))) .^ 2) : ring
    f = real.(ifft(fft(randn(rng, L, L)) .* filt))
    f = (f .- mean(f)) ./ std(f; corrected = false)
    img = Float32.(f .> (cls < 2 ? 0.0 : 0.85))
    return img .+ 0.15f0 .* randn(rng, Float32, L, L)
end

"주기 경계 쌍선형 회전 (scipy rotate(reshape = False, mode = 'wrap', order = 1)에 해당; 중심 기준)"
function rotate_wrap(img, deg)
    c = (L + 1) / 2; t = deg2rad(deg); out = similar(img)
    for j in 1:L, i in 1:L
        x = cos(t) * (i - c) + sin(t) * (j - c) + c
        yv = -sin(t) * (i - c) + cos(t) * (j - c) + c
        i0, j0 = floor(Int, x), floor(Int, yv); dx, dy = x - i0, yv - j0
        w(a, b) = img[mod1(a, L), mod1(b, L)]
        out[i, j] = (1 - dx) * (1 - dy) * w(i0, j0) + dx * (1 - dy) * w(i0 + 1, j0) + (1 - dx) * dy * w(i0, j0 + 1) + dx * dy * w(i0 + 1, j0 + 1)
    end
    return out
end

function dataset(n, rng, lam_range; aug = false)
    X = zeros(Float32, L, L, 1, n); y = zeros(Int, n)
    for i in 1:n
        c = (i - 1) % 3
        t = c == 0 ? lam_range[1] + (lam_range[2] - lam_range[1]) * rand(rng) : 0.0
        img = micro(c, rng; theta = t)
        aug && (img = rotate_wrap(img, 360 * rand(rng)))       # 회전 증강
        X[:, :, 1, i] = img; y[i] = c
    end
    return X, y
end

"circular padding을 두른 합성곱 층"
struct CConv{C}
    conv::C
    p::Int
end
Flux.@layer CConv
(c::CConv)(x) = c.conv(NNlib.pad_circular(x, (c.p, c.p, c.p, c.p); dims = (1, 2)))
cconv(rng, k, a, b, act = identity) = CConv(Conv((k, k), a => b, act; pad = 0, init = Flux.kaiming_uniform(rng; gain = sqrt(1 / 3))), k ÷ 2)

CNN(rng) = Chain(cconv(rng, 5, 1, 16, swish), MaxPool((2, 2)), cconv(rng, 5, 16, 32, swish), MaxPool((2, 2)),
                 cconv(rng, 3, 32, 32, swish), h -> dropdims(mean(h; dims = (1, 2)); dims = (1, 2)), tdense(rng, 32, 3))

function train(X, y, seed; epochs = 12)
    rng = Xoshiro(seed); net = CNN(rng)
    opt = Flux.setup(TorchAdamW(3e-3; lambda = 1e-2), net)
    for ep in 1:epochs, j in batches(rng, length(y), 64)
        xb = X[:, :, :, j]; yb = Flux.onehotbatch(y[j], 0:2)
        g = only(Flux.gradient(m -> Flux.logitcrossentropy(m(xb), yb), net))
        Flux.update!(opt, net, g)
    end
    return net
end

rot90(x) = reverse(permutedims(x, (2, 1, 3, 4)); dims = 1)    # 처음 두 축에서 90° 회전

function main()
    Xa, ya = dataset(1800, Xoshiro(0), (-15, 15))
    Xb, yb = dataset(1800, Xoshiro(0), (-15, 15); aug = true)
    angles = 0:15:90
    test = Dict(a => dataset(300, Xoshiro(100 + a), (a, a)) for a in angles)
    conv = cconv(Xoshiro(0), 5, 1, 8); x = Xa[:, :, :, 1:8]; fx = conv(x)
    e_t = norm(conv(circshift(x, (5, 11, 0, 0))) .- circshift(fx, (5, 11, 0, 0))) / norm(fx)
    e_r = norm(conv(rot90(x)) .- rot90(fx)) / norm(fx)
    @printf("random conv layer: relative error  translation (5,11) px: %.1e   rotation 90 deg: %.2f\n", e_t, e_r)
    for (name, (X, y)) in ("no augmentation" => (Xa, ya), "rotation augmentation" => (Xb, yb))
        R = zeros(2, length(angles), 2); tinv = 0.0
        for seed in 0:1
            net = train(X, y, seed)
            for (k, a) in enumerate(angles)
                Xt, yt = test[a]; p = Flux.onecold(net(Xt), 0:2); lam = yt .== 0
                R[seed+1, k, :] = [mean(p[lam] .== 0), mean(p[.!lam] .== yt[.!lam])]
            end
            Xt, _ = test[0]
            tinv = maximum(abs.(net(Xt) .- net(circshift(Xt, (13, 29, 0, 0)))))
        end
        R = dropdims(mean(R; dims = 1); dims = 1)
        @printf("--- %s (logit change under (13,29)-px roll: %.1e)\n", name, tinv)
        println("   lamella angle (deg): ", join(lpad.(collect(angles), 7)))
        println("   lamellae recall    : ", join([@sprintf("%7.3f", v) for v in R[:, 1]]))
        println("   other two classes  : ", join([@sprintf("%7.3f", v) for v in R[:, 2]]))
    end
end

main()
