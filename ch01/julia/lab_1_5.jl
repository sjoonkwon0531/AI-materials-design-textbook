# Lab 1.5 (Julia): Truncated SVD로 보는 정규화 — 인구 데이터 2차 회귀
using LinearAlgebra, Printf, Statistics

yr  = Float64[1930, 1940, 1949, 1960, 1970, 1980, 1990, 2000, 2010]
pop = Float64[2044, 2355, 2017, 2499, 3144, 3741, 4339, 4599, 4799]
A = hcat(yr .^ 2, yr, ones(length(yr))); B = pop

F = svd(A)              # thin SVD: A = U * Diagonal(S) * V'  (NumPy의 Vt가 아니라 V를 돌려준다)
U, s, V = F.U, F.S, F.V
println("singular values: ", join([@sprintf("%.4g", si) for si in s], ", "))  # 1.165e+07, 77.68, 0.0004508
@printf("cond(A) = s1/s_min: %.3g\n", s[1] / s[end])                         # 2.58e+10

function fit(t)
    sinv = [i <= t ? 1 / s[i] : 0.0 for i in eachindex(s)]   # 1-based: 처음 t개만 유지
    X = V * Diagonal(sinv) * U' * B
    rms = sqrt(mean((A * X - B) .^ 2))
    return X, rms
end

for t in (3, 2, 1)
    X, rms = fit(t)
    @printf("t=%d: a=%.4g b=%.4g c=%.4g  RMS=%.2f\n", t, X[1], X[2], X[3], rms)
end
