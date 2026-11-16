# 연습문제 1.3.E: FP32 vs FP64 신경망 학습과 오차 3성분 분해
# 해석: x in [-2,2], 학습점 N=128 (균일 랜덤), 노이즈 sigma=0.05, 학습셋 R=8개
#       MLP 1-64-64-1 (tanh), Adam lr=3e-3, full-batch 4000 step
#       같은 데이터·같은 초기 가중치로 FP64/FP32를 각각 학습 → 정밀도 성분 분리
import numpy as np, torch, time
torch.set_num_threads(1)
f = lambda x: np.sin(5*x)*np.cos(x**2)
R, N, sigma, steps = 8, 128, 0.05, 4000
xt = np.linspace(-2, 2, 801)
def make_net(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1,64), torch.nn.Tanh(), torch.nn.Linear(64,64), torch.nn.Tanh(), torch.nn.Linear(64,1)).double()
def train(net0, x, y, dtype):
    net = make_net(0); net.load_state_dict(net0.state_dict()); net = net.to(dtype)
    X = torch.tensor(x[:,None], dtype=dtype); Y = torch.tensor(y[:,None], dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad(); loss = ((net(X)-Y)**2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor(xt[:,None], dtype=dtype)).double().numpy().ravel(), loss.item()
t0 = time.time(); P64, P32 = [], []
for r in range(R):
    rng = np.random.default_rng(100+r)
    x = rng.uniform(-2, 2, N); y = f(x) + rng.normal(0, sigma, N)
    net0 = make_net(1000+r)
    p64, l64 = train(net0, x, y, torch.float64); p32, l32 = train(net0, x, y, torch.float32)
    P64.append(p64); P32.append(p32)
    print(f"r={r}: final train MSE FP64={l64:.3e} FP32={l32:.3e}  max|p32-p64|={np.abs(p32-p64).max():.2e}")
P64, P32 = np.array(P64), np.array(P32)
Fbar = P64.mean(0)
bias2 = (Fbar - f(xt))**2
var = P64.var(0, ddof=1)
prec = ((P32 - P64)**2).mean(0)
regions = {"|x|<1": np.abs(xt) < 1, "1<=|x|<1.75": (np.abs(xt) >= 1) & (np.abs(xt) < 1.75), "|x|>=1.75": np.abs(xt) >= 1.75}
for k, m in regions.items():
    print(f"{k:12s}: Bias^2={bias2[m].mean():.2e}  Var={var[m].mean():.2e}  Prec(FP32-FP64)^2={prec[m].mean():.2e}")
print(f"overall    : Bias^2={bias2.mean():.2e}  Var={var.mean():.2e}  Prec={prec.mean():.2e}  sigma^2={sigma**2:.2e}")
# 정밀도 성분이 '반올림 그 자체'인지 '궤적 분기'인지: 학습 0 step 후 차이(순전파만) 비교
net0 = make_net(1000)
with torch.no_grad():
    a = net0(torch.tensor(xt[:,None])).numpy().ravel()
    b = net0.float()(torch.tensor(xt[:,None], dtype=torch.float32)).double().numpy().ravel()
print(f"forward-only FP32 vs FP64 rms diff (untrained net) = {np.sqrt(np.mean((a-b)**2)):.2e}")
print(f"after training rms(P32-P64) = {np.sqrt(prec.mean()):.2e}")
print(f"elapsed {time.time()-t0:.0f}s")
np.savez("ex_1_3_E.npz", xt=xt, P64=P64, P32=P32)
