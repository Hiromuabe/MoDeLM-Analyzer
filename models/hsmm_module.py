import torch
import torch.nn.functional as F
import numpy as np

EPS = 1e-12
LOGEPS = -1e30

# --------- utility ---------
def _logsumexp_torch(a, axis=None):
    if axis is None:
        amax = torch.max(a)
        if torch.isinf(amax): 
            return amax
        return amax + torch.log(torch.sum(torch.exp(torch.clamp(a - amax, max=50))))
    else:
        amax, _ = torch.max(a, dim=axis, keepdim=True)
        amax = torch.where(torch.isinf(amax), torch.zeros_like(amax), amax)
        out = amax + torch.log(torch.sum(torch.exp(torch.clamp(a - amax, max=50)), dim=axis, keepdim=True))
        return out.squeeze(axis)

def _spd_regularize_torch(S, jitter=1e-1, max_tries=3):
    S = 0.5*(S + S.transpose(-1,-2))
    eye = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    for i in range(max_tries):
        try:
            w, V = torch.linalg.eigh(S)
            w = torch.clamp(w, min=jitter)
            return (V * w.unsqueeze(0)) @ V.transpose(-1, -2)
        except Exception:
            S = S + (10**i) * jitter * eye
    return jitter * eye


# --------- Hybrid Initialization ---------
def _hybrid_init(Y_np, K, seed=0, noise_scale=0.1):
    """
    KMeansで大まかなクラスタを取得し、適度なノイズで分散させる
    - 再現性: KMeansは決定的（seed固定）
    - 多様性: ノイズでseed依存の多様性を追加
    """
    from sklearn.cluster import KMeans
    
    # 固定seedでKMeansを実行（再現性）
    km = KMeans(n_clusters=K, n_init=20, random_state=0, max_iter=500).fit(Y_np)
    base_centers = km.cluster_centers_
    labels = km.labels_
    
    # seedに応じてノイズを追加（多様性）
    rng = np.random.RandomState(seed)
    data_scale = np.std(Y_np, axis=0).mean()
    noise = rng.randn(K, Y_np.shape[1]) * data_scale * noise_scale
    centers = base_centers + noise
    
    return centers, labels


def _robust_covariance(data, global_cov, min_samples=5):
    """ロバストな共分散推定（外れ値に強い）"""
    if len(data) < min_samples:
        return global_cov
    
    # 中央値ベースの正規化で外れ値の影響を減らす
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    mad = np.where(mad < 1e-6, 1.0, mad)
    normalized = (data - median) / mad
    
    # 外れ値を除外（3-sigma rule）
    distances = np.sqrt(np.sum(normalized**2, axis=1))
    mask = distances < 3.0
    
    if mask.sum() < min_samples:
        mask = np.ones(len(data), dtype=bool)
    
    filtered_data = data[mask]
    cov = np.cov(filtered_data.T)
    
    # 正則化
    cov = cov + 0.05 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    
    return cov


# --------- HSMM-LDS ---------
class HSMM_LDS_Torch:
    def __init__(self, K, Dmax, obs_dim, init_mean_dur=20.0,
                 allow_self_transition=True, duration_model="poisson",
                 device="cpu", dtype=torch.float32, Y_init=None, seed=0):
        self.K = K
        self.Dmax = Dmax
        self.r = obs_dim
        self.device = device
        self.dtype = dtype
        self.duration_model = duration_model

        # 遷移行列
        if allow_self_transition:
            self.A = 0.95*torch.eye(K, device=device) + \
                     0.05/(K-1)*(torch.ones((K,K), device=device)-torch.eye(K, device=device))
        else:
            A = torch.full((K, K), 1.0/(K-1), device=device)
            A.fill_diagonal_(0.0)
            self.A = A
        self.pi = torch.full((K,), 1.0/K, device=device)

        # --- ハイブリッド初期化 ---
        if Y_init is not None:
            Y_np = Y_init.cpu().numpy()
            
            # データのスケール情報
            data_std = np.std(Y_np, axis=0).mean()
            
            # ハイブリッド初期化: KMeans + seed依存ノイズ
            centers, labels = _hybrid_init(Y_np, K, seed=seed, noise_scale=0.15)
            
            # 診断情報
            cluster_counts = np.bincount(labels)
            print(f"[Init seed={seed}] Cluster sizes: {cluster_counts}")
            print(f"[Init seed={seed}] Cluster balance: min={cluster_counts.min()}, max={cluster_counts.max()}")
            
            self.bk = torch.tensor(centers, device=device, dtype=dtype)
            self.Ak = torch.stack([torch.eye(obs_dim, device=device, dtype=dtype) for _ in range(K)])
            
            # 各クラスタの共分散をロバストに推定
            global_cov = np.cov(Y_np.T) + 0.05 * np.eye(obs_dim) * (data_std ** 2)
            
            self.Sig = []
            for k in range(K):
                cluster_pts = Y_np[labels == k]
                
                if len(cluster_pts) >= 5:
                    cov = _robust_covariance(cluster_pts, global_cov)
                else:
                    cov = global_cov
                
                # スケール正規化
                scale_factor = np.trace(cov) / obs_dim
                target_scale = data_std ** 2
                cov = cov * (target_scale / (scale_factor + 1e-6))
                
                self.Sig.append(torch.tensor(cov, device=device, dtype=dtype))
            
            self.Sig = torch.stack(self.Sig)
            
        else:
            # fallback
            self.Ak = torch.stack([torch.eye(obs_dim, device=device, dtype=dtype) for _ in range(K)])
            self.bk = torch.zeros((K, obs_dim), device=device, dtype=dtype)
            self.Sig = torch.stack([torch.eye(obs_dim, device=device, dtype=dtype) for _ in range(K)])

        # Duration parameters
        if duration_model == "geom":
            q0 = 1.0 / init_mean_dur
            self.q = torch.full((K,), q0, device=device, dtype=dtype)
        elif duration_model == "poisson":
            self.lam = torch.full((K,), init_mean_dur, device=device, dtype=dtype)
        elif duration_model == "negbin":
            mean_d = init_mean_dur
            var_d = 2*mean_d
            p0 = mean_d/var_d
            r0 = mean_d*p0/(1-p0)
            self.p_nb = torch.full((K,), p0, device=device, dtype=dtype)
            self.r_nb = torch.full((K,), r0, device=device, dtype=dtype)

    # ---- emission ----
    def _log_emission_per_t(self, Y):
        T, r = Y.shape
        T1 = T-1
        LE = torch.full((T1,self.K), LOGEPS, device=self.device, dtype=self.dtype)
        for k in range(self.K):
            pred = (Y[:-1] @ self.Ak[k].T) + self.bk[k]
            diff = Y[1:] - pred
            Sigk = _spd_regularize_torch(self.Sig[k], jitter=0.1)
            invS = torch.inverse(Sigk)
            logdet = torch.logdet(Sigk + 1e-3*torch.eye(r, device=self.device))
            quad = torch.sum((diff @ invS) * diff, dim=1)
            LE[:,k] = -0.5*(quad + r*torch.log(torch.tensor(2*3.14159,device=self.device)) + logdet)
        return LE

    # ---- block emission ----
    def _log_block_emission(self, LE):
        T1, K = LE.shape
        ps = torch.zeros((T1+1, K), device=self.device, dtype=self.dtype)
        ps[1:] = torch.cumsum(LE, dim=0)
        
        BL = torch.full((T1, self.Dmax, K), LOGEPS, device=self.device, dtype=self.dtype)
        for t in range(T1):
            maxd = min(self.Dmax, t+1)
            for d in range(1, maxd+1):
                BL[t, d-1, :] = ps[t+1, :] - ps[t+1-d, :]
        return BL

    # ---- duration pmf ----
    def _log_dur_pmf(self):
        d = torch.arange(1,self.Dmax+1,device=self.device,dtype=self.dtype).unsqueeze(1)
        if self.duration_model=="geom":
            q = torch.clamp(self.q[None,:],1e-6,1-1e-6)
            logpmf = (d-1)*torch.log1p(-q) + torch.log(q)
        elif self.duration_model=="poisson":
            lam = torch.clamp(self.lam[None,:],1e-3,1000)
            logpmf = d*torch.log(lam) - lam - torch.lgamma(d+1)
        elif self.duration_model=="negbin":
            r = torch.clamp(self.r_nb[None,:],1e-2,1e3)
            p = torch.clamp(self.p_nb[None,:],1e-3,1-1e-3)
            logpmf = (torch.lgamma(d+r-1)-torch.lgamma(d)-torch.lgamma(r)) \
                     + r*torch.log1p(-p) + (d-1)*torch.log(p)
        logZ = _logsumexp_torch(logpmf,axis=0)
        return logpmf - logZ

    # ---- Forward-Backward ----
    def _forward_backward(self, Y):
        T, r = Y.shape
        T1 = T - 1
        K, D = self.K, self.Dmax
        
        LE = self._log_emission_per_t(Y)
        BL = self._log_block_emission(LE)
        LD = self._log_dur_pmf()
        logA = torch.log(torch.clamp(self.A, min=EPS))
        
        # Forward
        alpha_end = torch.full((T1, K), LOGEPS, device=self.device, dtype=self.dtype)
        for t in range(T1):
            maxd = min(D, t+1)
            for k in range(K):
                terms = []
                for d in range(1, maxd+1):
                    block = BL[t, d-1, k] + LD[d-1, k]
                    if d == t+1:
                        prior = torch.log(self.pi[k] + EPS)
                    else:
                        prior = _logsumexp_torch(alpha_end[t-d, :] + logA[:, k])
                    terms.append(prior + block)
                alpha_end[t, k] = _logsumexp_torch(torch.stack(terms))
        
        logZ = _logsumexp_torch(alpha_end[T1-1, :])
        
        # Backward
        beta_end = torch.zeros((T1, K), device=self.device, dtype=self.dtype)
        for t in range(T1-2, -1, -1):
            maxd2 = min(D, (T1-1) - t)
            if maxd2 <= 0:
                continue
            
            next_terms = torch.full((K,), LOGEPS, device=self.device, dtype=self.dtype)
            for l in range(K):
                vals = []
                for d2 in range(1, maxd2+1):
                    u = t + d2
                    block = BL[u, d2-1, l] + LD[d2-1, l]
                    future = beta_end[u, l] if u < T1-1 else 0.0
                    vals.append(block + future)
                next_terms[l] = _logsumexp_torch(torch.stack(vals))
            
            for k in range(K):
                beta_end[t, k] = _logsumexp_torch(logA[k, :] + next_terms)
        
        # Segment posterior
        post_seg = torch.full((T1, D, K), LOGEPS, device=self.device, dtype=self.dtype)
        for t in range(T1):
            maxd = min(D, t+1)
            for k in range(K):
                for d in range(1, maxd+1):
                    block = BL[t, d-1, k] + LD[d-1, k]
                    if d == t+1:
                        prior = torch.log(self.pi[k] + EPS)
                    else:
                        prior = _logsumexp_torch(alpha_end[t-d, :] + logA[:, k])
                    future = beta_end[t, k]
                    post_seg[t, d-1, k] = prior + block + future - logZ
        
        # Transition expectations
        exp_trans = torch.zeros((K, K), device=self.device, dtype=self.dtype)
        for t in range(T1):
            maxd = min(D, t+1)
            for j in range(K):
                for d in range(1, maxd+1):
                    w = torch.exp(post_seg[t, d-1, j])
                    if w < 1e-15 or d == t+1:
                        continue
                    prev_logits = alpha_end[t-d, :] + logA[:, j]
                    prev_logZ = _logsumexp_torch(prev_logits)
                    prev_prob = torch.exp(prev_logits - prev_logZ)
                    exp_trans[:, j] += w * prev_prob
        
        # Gamma (state occupation)
        W = torch.exp(post_seg)
        W_rev = torch.flip(W, dims=[1])
        W_cumd = torch.cumsum(W_rev, dim=1)
        W_tail = torch.flip(W_cumd, dims=[1])
        
        gamma = torch.zeros((T1, K), device=self.device, dtype=self.dtype)
        for u in range(T1):
            maxd = min(D, u+1)
            if maxd <= 0:
                continue
            dmins = torch.arange(1, maxd+1, device=self.device)
            t_idx = u - dmins + 1
            mask = t_idx >= 0
            valid_t = t_idx[mask].long()
            valid_d = dmins[mask].long()
            gamma[valid_t, :] += W_tail[u, valid_d-1, :]
        
        s = gamma.sum(dim=1, keepdim=True)
        s = torch.where(s < EPS, torch.ones_like(s), s)
        gamma_tk = gamma / s
        
        # Duration expectations
        exp_dur = torch.zeros(K, device=self.device, dtype=self.dtype)
        for k in range(K):
            wsum = 0.0
            dsum = 0.0
            for t in range(T1):
                maxd = min(D, t+1)
                if maxd <= 0:
                    continue
                w = torch.exp(post_seg[t, :maxd, k])
                dvals = torch.arange(1, maxd+1, device=self.device, dtype=self.dtype)
                dsum += (w * dvals).sum()
                wsum += w.sum()
            exp_dur[k] = dsum / max(wsum, EPS)
        
        return gamma_tk, exp_trans, exp_dur, logZ

    # ---- M-step with smoothing ----
    def _m_step(self, Y, gamma_tk, exp_trans, exp_dur):
        T, r = Y.shape
        T1 = T - 1
        
        # Update A with smoothing
        row_sums = exp_trans.sum(dim=1, keepdim=True)
        A_new = exp_trans / (row_sums + EPS)
        self.A = 0.8 * A_new + 0.2 * self.A
        
        # Regression update
        X = torch.cat([Y[:-1], torch.ones(T1, 1, device=self.device)], dim=1)
        Yn = Y[1:]
        
        for k in range(self.K):
            w = torch.clamp(gamma_tk[:, k], min=0.0)
            wsum = w.sum()
            
            if wsum < 1e-3:
                continue
            
            sw = torch.sqrt(w)[:, None]
            Xw, Yw = X * sw, Yn * sw
            
            # Ridge regression
            reg = 0.02 * (wsum / T1)
            XtX = Xw.T @ Xw + reg * torch.eye(Xw.shape[1], device=self.device, dtype=self.dtype)
            XtY = Xw.T @ Yw
            
            try:
                theta = torch.linalg.solve(XtX, XtY)
            except:
                theta = torch.linalg.pinv(Xw) @ Yw
            
            # Smooth update
            Ak_new = theta[:-1, :].T
            bk_new = theta[-1, :]
            
            self.Ak[k] = 0.8 * Ak_new + 0.2 * self.Ak[k]
            self.bk[k] = 0.8 * bk_new + 0.2 * self.bk[k]
            
            # Covariance
            resid = Yn - X @ theta
            Sk = (resid * sw).T @ (resid * sw) / (wsum + EPS)
            
            # Regularize
            diag_mean = torch.diag(Sk).mean()
            Sk = Sk + 0.15 * diag_mean * torch.eye(r, device=self.device, dtype=self.dtype)
            
            try:
                Sig_new = _spd_regularize_torch(Sk, jitter=0.12)
                self.Sig[k] = 0.8 * Sig_new + 0.2 * self.Sig[k]
            except Exception:
                pass
        
        # Duration update with smoothing and strict bounds
        if self.duration_model == "geom":
            q_new = 1.0 / torch.clamp(exp_dur, min=2.5, max=self.Dmax/2)
            self.q = 0.75 * q_new + 0.25 * self.q
        elif self.duration_model == "poisson":
            lam_new = torch.clamp(exp_dur, min=2.5, max=self.Dmax / 2)
            self.lam = 0.75 * lam_new + 0.25 * self.lam
        elif self.duration_model == "negbin":
            mu = torch.clamp(exp_dur, min=2.5, max=self.Dmax / 2)
            var = mu * 2
            var = torch.maximum(var, mu + 1e-6)
            var = torch.minimum(var, torch.tensor(self.Dmax**2, device=self.device, dtype=self.dtype))
            p_star = mu / var
            r_star = mu * p_star / (1 - p_star + 1e-6)
            self.p_nb = 0.8 * self.p_nb + 0.2 * p_star
            self.r_nb = 0.8 * self.r_nb + 0.2 * r_star

    # ---- Training ----
    def fit(self, Y, n_iter=10, verbose=True):
        Y = Y.to(self.device, dtype=self.dtype)
        last_ll = -1e9
        
        for it in range(n_iter):
            gamma_tk, exp_trans, exp_dur, logZ = self._forward_backward(Y)
            
            if verbose:
                state_usage = gamma_tk.sum(dim=0)
                active_states = (state_usage > 1.0).sum().item()
                print(f"[EM {it}] log p(Y) = {logZ.item():.3f}, active states: {active_states}/{self.K}")
                print(f"  State usage: {state_usage.cpu().numpy()}")
                print(f"  exp_dur: {exp_dur.cpu().numpy()}")
            
            self._m_step(Y, gamma_tk, exp_trans, exp_dur)
            
            if abs(logZ - last_ll) < 1e-3:
                if verbose:
                    print(f"Converged at iteration {it}")
                break
            last_ll = logZ
        
        return logZ

    # ---- Viterbi decode ----
    def viterbi_decode(self, Y):
        Y = Y.to(self.device, dtype=self.dtype)
        T, r = Y.shape
        T1 = T - 1
        K, D = self.K, self.Dmax
        
        LE = self._log_emission_per_t(Y)
        BL = self._log_block_emission(LE)
        LD = self._log_dur_pmf()
        logA = torch.log(torch.clamp(self.A, min=EPS))
        
        V = torch.full((T1, K), LOGEPS, device=self.device)
        BP_d = torch.full((T1, K), -1, device=self.device, dtype=torch.long)
        BP_k = torch.full((T1, K), -1, device=self.device, dtype=torch.long)
        
        for t in range(T1):
            maxd = min(D, t + 1)
            for k in range(K):
                best = LOGEPS
                bd, bk = -1, -1
                for d in range(1, maxd + 1):
                    block = BL[t, d-1, k] + LD[d-1, k]
                    if d == t + 1:
                        cand = torch.log(self.pi[k] + EPS) + block
                        prevk = -1
                    else:
                        prev = V[t-d, :] + logA[:, k]
                        j = torch.argmax(prev)
                        cand = prev[j] + block
                        prevk = j.item()
                    
                    if cand > best:
                        best, bd, bk = cand, d, prevk
                
                V[t, k] = best
                BP_d[t, k] = bd
                BP_k[t, k] = bk
        
        t = T1 - 1
        k = torch.argmax(V[t, :]).item()
        z = torch.empty(T1, dtype=torch.long, device=self.device)
        
        while t >= 0 and k >= 0:
            d = BP_d[t, k].item()
            s = t - d + 1
            z[s:t+1] = k
            k = BP_k[t, k].item()
            t = s - 1
        
        return z.cpu()

# --------- helpers ---------
def temporal_pool(Z, w=5):
    pad = (w-1)//2
    left = Z[0].unsqueeze(0).repeat(pad, 1)
    right = Z[-1].unsqueeze(0).repeat(pad, 1)
    Zp = torch.cat([left, Z, right], dim=0)
    return torch.stack([Zp[i:i+len(Z)] for i in range(w)], 0).mean(0)

def zscore_torch(Y, eps=1e-8):
    mu = Y.mean(dim=0, keepdim=True)
    sd = Y.std(dim=0, keepdim=True).clamp_min(eps)
    return (Y - mu) / sd
