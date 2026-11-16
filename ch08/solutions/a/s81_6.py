"""Ex 8.1.6: GAE lambda map (critic 1500 vs 100 iters) and PPO-clip vs A2C on CartPole."""
import sys, importlib, time, numpy as np, torch
sys.path.insert(0, "../../../ch08/labs"); torch.set_num_threads(1)
L = importlib.import_module("81_pg")
import gymnasium as gym
LAMS = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]

def lam_map(pi, iters_list=(1500, 100), n_fit=300, n_eval=300, seed=123):
    env = gym.make("CartPole-v1")
    data = [L.rollout(env, pi, seed=seed * 100_000 + i) for i in range(n_fit)]
    Sx = torch.cat([d[0] for d in data]); Gy = torch.as_tensor(np.concatenate([L.reward_to_go(d[2]) for d in data]))
    eps = [L.rollout(env, pi, seed=seed * 100_000 + n_fit + i) for i in range(n_eval)]
    print("mean len fit", np.mean([len(d[2]) for d in data]), "eval", np.mean([len(e[2]) for e in eps]),
          "terminated frac", np.mean([e[3] for e in eps]))
    exact = np.array([L.flat_grad(pi, S, A, L.GAMMA ** np.arange(len(rw)) * L.reward_to_go(rw)) for S, A, rw, _, _ in eps])
    rtg = np.array([L.flat_grad(pi, S, A, L.reward_to_go(rw)) for S, A, rw, _, _ in eps])
    ref0 = exact.mean(0); base = rtg.var(0, ddof=1).sum()
    out = {}
    for iters in iters_list:
        torch.manual_seed(seed); Vf = L.mlp(1); opt = torch.optim.Adam(Vf.parameters(), lr=1e-2)
        for it in range(iters):
            opt.zero_grad(); loss = ((Vf(Sx).squeeze(-1) - Gy) ** 2).mean(); loss.backward(); opt.step()
        # critic error on eval episodes (vs MC return)
        with torch.no_grad():
            Se = torch.cat([e[0] for e in eps]); Ge = np.concatenate([L.reward_to_go(e[2]) for e in eps])
            ve = Vf(Se).squeeze(-1).numpy()
        rmse = np.sqrt(np.mean((ve - Ge) ** 2)); bias = np.mean(ve - Ge)
        print(f"critic iters {iters}: train MSE {loss.item():.1f}, eval RMSE vs G_t {rmse:.2f}, mean(V-G) {bias:+.2f}, mean G {Ge.mean():.1f}")
        rows = []
        gv = []
        for S, A, rw, term, sl in eps:
            with torch.no_grad():
                v = Vf(S).squeeze(-1).numpy(); vl = 0.0 if term else float(Vf(sl))
            gv.append(L.flat_grad(pi, S, A, L.reward_to_go(rw) - v))
        gv = np.array(gv); m = gv.mean(0)
        rows.append(("G-V", gv.var(0, ddof=1).sum() / base, np.linalg.norm(m), m @ ref0 / np.linalg.norm(m) / np.linalg.norm(ref0)))
        for lam in LAMS:
            g = []
            for S, A, rw, term, sl in eps:
                with torch.no_grad():
                    v = Vf(S).squeeze(-1).numpy(); vl = 0.0 if term else float(Vf(sl))
                g.append(L.flat_grad(pi, S, A, L.gae(rw, v, vl, lam)))
            g = np.array(g); m = g.mean(0)
            rows.append((f"lam={lam}", g.var(0, ddof=1).sum() / base, np.linalg.norm(m), m @ ref0 / np.linalg.norm(m) / np.linalg.norm(ref0)))
        for r in rows: print(f"   {r[0]:9s} var/G_t {r[1]:.4f}  |mean| {r[2]:7.1f}  cos(exact) {r[3]:+.3f}")
        out[iters] = rows
    print(f"   exact |mean| {np.linalg.norm(ref0):.1f}, reward-to-go var ref {base:.3g}")
    env.close(); return out

def train_ppo(seed, n_ep=300, lr_pi=3e-3, lr_v=1e-2, clip=0.2, n_batch=4, n_epoch=4):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make("CartPole-v1"); env.action_space.seed(seed)
    pi, V = L.mlp(2), L.mlp(1)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=lr_pi); opt_v = torch.optim.Adam(V.parameters(), lr=lr_v)
    lengths = []; clipped = []; ep = 0
    while ep < n_ep:
        batch = []
        for _ in range(n_batch):
            if ep >= n_ep: break
            S, A, rw, term, sl = L.rollout(env, pi, seed=seed * 10_000 + ep); ep += 1
            lengths.append(len(rw)); batch.append((S, A, rw, term, sl))
        Ss, As, Advs, Tgts = [], [], [], []
        with torch.no_grad():
            for S, A, rw, term, sl in batch:
                v = V(S).squeeze(-1).numpy(); vl = 0.0 if term else float(V(sl))
                Adv = L.gae(rw, v, vl); Ss.append(S); As.append(A); Advs.append(Adv); Tgts.append(Adv + v)
        S = torch.cat(Ss); A = torch.cat(As)
        Adv = torch.as_tensor(np.concatenate(Advs)); Tg = torch.as_tensor(np.concatenate(Tgts))
        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)
        with torch.no_grad():
            logp_old = torch.distributions.Categorical(logits=pi(S)).log_prob(A)
        for k in range(n_epoch):
            logp = torch.distributions.Categorical(logits=pi(S)).log_prob(A)
            ratio = torch.exp(logp - logp_old)
            obj = torch.min(ratio * Adv, torch.clamp(ratio, 1 - clip, 1 + clip) * Adv)
            clipped.append(((ratio - 1).abs() > clip).float().mean().item())
            opt_pi.zero_grad(); (-obj.mean()).backward(); opt_pi.step()
            for _ in range(5):
                opt_v.zero_grad(); ((V(S).squeeze(-1) - Tg) ** 2).mean().backward(); opt_v.step()
    env.close()
    return np.array(lengths), np.array(clipped)

def summarize(name, Ls):
    last = np.array([l[-50:].mean() for l in Ls]); f475 = []; coll = []
    for l in Ls:
        ma = np.convolve(l, np.ones(20) / 20, "valid")
        if (ma >= 475).any():
            i = int(np.argmax(ma >= 475)); f475.append(i + 20)
            coll.append(bool((ma[i:] < 300).any()))
        else: f475.append(-1); coll.append(None)
    print(f"{name}: last50 {last.mean():.1f} +- {last.std(ddof=1)/np.sqrt(len(Ls)):.1f} per seed {np.round(last).astype(int).tolist()} | first475 {f475} | collapse(<300 after 475) {coll}")

if __name__ == "__main__":
    t0 = time.time(); part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("all", "lam"):
        _, pi_mid, _ = L.train("baseline", 0, n_ep=60)
        lam_map(pi_mid); print(f"[{time.time()-t0:.0f} s]"); sys.stdout.flush()
    if part in ("all", "ppo"):
        Ls_a2c = []; Ls_ppo = []; clips = []
        for sd in range(5):
            l, _, _ = L.train("a2c", sd); Ls_a2c.append(l)
            l, c = train_ppo(sd); Ls_ppo.append(l); clips.append(c)
            print(f" seed {sd} done [{time.time()-t0:.0f} s] ppo clipfrac mean {c.mean():.3f} (epoch1 {c[0::4].mean():.3f}, epoch4 {c[3::4].mean():.3f})"); sys.stdout.flush()
        summarize("A2C", Ls_a2c); summarize("PPO", Ls_ppo)
        allc = np.concatenate(clips); print(f"PPO clip fraction overall {allc.mean():.3f}; epochs 2-4 only {np.concatenate([c.reshape(-1,4)[:,1:].ravel() for c in clips]).mean():.3f}")
        np.savez("ppo_a2c.npz", a2c=np.array(Ls_a2c), ppo=np.array(Ls_ppo))
    print(f"wall {time.time()-t0:.0f} s")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "noclip":
    Ls = []; t0 = time.time()
    for sd in range(5):
        l, c = train_ppo(sd, clip=1e9); Ls.append(l)
    summarize("PPO without clipping (same batching, normalization, 4 epochs)", Ls)
    Ls = []
    for sd in range(5):
        l, c = train_ppo(sd, n_epoch=1); Ls.append(l)
    summarize("PPO 1 epoch (clip inactive)", Ls)
    print(f"wall {time.time()-t0:.0f} s")
