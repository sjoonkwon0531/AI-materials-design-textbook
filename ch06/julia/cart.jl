# cart.jl — 깊이를 제한한 이진 분류 CART(Gini, 문턱 = 이웃한 두 값의 중점: sklearn DecisionTreeClassifier와 같은 규칙)
# 작은 표(Lab 6.19, 6.20)에서 sklearn의 문턱과 정확도를 그대로 재현하려고 직접 구현했다.
# 동률 분기(이득이 정확히 같은 두 후보)가 있으면 먼저 찾은 특징·위치를 고른다(sklearn은 특징 순서를 섞는다).
gini(n1, n) = n == 0 ? 0.0 : 2 * (n1 / n) * (1 - n1 / n)

"노드: 잎이면 (p1,), 아니면 (특징, 문턱, 왼쪽, 오른쪽)"
function cart_fit(X, y; max_depth = 3, depth = 0)
    n = length(y); n1 = sum(y); p1 = n1 / n
    (depth >= max_depth || n1 == 0 || n1 == n) && return (p1,)
    best = (Inf, 0, 0.0)
    for j in 1:size(X, 2)
        o = sortperm(X[:, j]); xs = X[o, j]; c1 = cumsum(y[o])
        for i in 1:n-1
            xs[i+1] > xs[i] || continue
            l1 = c1[i]; r1 = n1 - l1
            imp = (i * gini(l1, i) + (n - i) * gini(r1, n - i)) / n
            imp < best[1] && (best = (imp, j, 0.5 * (xs[i] + xs[i+1])))
        end
    end
    best[2] == 0 && return (p1,)
    _, j, t = best; L = X[:, j] .<= t
    return (j, t, cart_fit(X[L, :], y[L]; max_depth, depth = depth + 1), cart_fit(X[.!L, :], y[.!L]; max_depth, depth = depth + 1))
end
cart_proba(node, x) = length(node) == 1 ? node[1] : cart_proba(x[node[1]] <= node[2] ? node[3] : node[4], x)
cart_predict(node, X) = [cart_proba(node, r) > 0.5 for r in eachrow(X)]
cart_thresholds(node) = length(node) == 1 ? Float64[] : vcat(node[2], cart_thresholds(node[3]), cart_thresholds(node[4]))
