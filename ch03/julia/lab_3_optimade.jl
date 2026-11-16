# OPTIMADE 질의 (Julia) — 여러 DB를 한 질의로 (키 불필요; 네트워크 필요)
# 파이썬 판(requests)과 같은 필터를 세 제공자에 그대로 던진다. HTTP.jl은 2xx가 아니면
# 예외를 던지므로(status_exception=true 기본값) requests의 raise_for_status()와 같은 역할을 한다.
#   준비:  ] add HTTP JSON3
using HTTP, JSON3

providers = [   # 순서를 고정하려고 Dict 대신 Pair 벡터를 쓴다(Julia Dict는 순서를 보장하지 않음)
    "Materials Project" => "https://optimade.materialsproject.org/v1/structures",
    "OQMD"              => "https://oqmd.org/optimade/v1/structures",
    "JARVIS"            => "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures",
]
flt = "chemical_formula_reduced=\"O2Si\""
for (name, base) in providers
    r = HTTP.get(base; query = ["filter" => flt, "page_limit" => "5"], readtimeout = 30)
    js = JSON3.read(r.body)
    println(name, " ", js.meta.data_returned)
end
