"""River Network Topology Recovery from Pulse Traces - end-to-end solution.

usage: python3 solution.py <public_dataset_dir> <submission_csv_path>

Requirement map
- Combine the windows of each opaque network_id: every pairwise feature is a window-count-independent
  statistic (mean / fraction) over all windows of one network (test: its 28 windows); one tree is decoded
  per network and written to each of its rows.
- Slot-invariant relational signal only (slot names are never features): lagged co-movement of
  discharge_level / discharge_change / network_discharge_level (lagcorr), rainfall-conditioned response
  timing and rainfall similarity (group_rain), rising-limb event timing (group_events), confluence evidence
  from non-negative least-squares fits of a channel's flow by lagged upstream-candidate flows plus own
  rainfall (group_nnls), network-scale magnitude differences, and within-row scale / variability ranks.
- Learned inference rule: a LightGBM model with a per-node softmax objective (softmax_objective,
  train_model) is trained in this script on the training networks; for each channel it scores every other
  channel as its immediate downstream child plus an "outlet" option.
- Window-count robustness: training examples are built from random 5-window subsets of each training
  network (SUB_K windows x SUB_DRAWS draws, fixed RandomState), so the model learns from evidence of
  varying quality; test networks use all their windows.
- Globally consistent directed tree: decode_tree() returns the maximum-likelihood rooted tree under the
  learned head log-probabilities by exact search over all rooted labelled trees on the visible channels
  (enumerate_trees): channel_count-1 edges, one outlet, no cycles, complete parent groups.
- Canonical program: to_program() emits PARENT[+PARENT...]>CHILD tokens with sorted parents, ordered by
  depth then token string, single spaces.
- Only the supplied public files are used; test rows are used only for inference (no fitting on test).
- Determinism: fixed seeds, fixed thread count (THREADS), deterministic LightGBM, no clock reads.
- CPU only. Measured runtime: see README (a few minutes on 4 CPU cores).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls

THREADS = 4
SEED = 0
ROUNDS = 1000
SUB_K = 5
SUB_DRAWS = 4
LAGS = list(range(-6, 13))
SIGS = ['discharge_level', 'network_discharge_level', 'discharge_change', 'rainfall_level']
PARAMS = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=THREADS, seed=SEED,
              deterministic=True, force_col_wise=True)


# ----------------------------------------------------------------------------- data
def slot_name(i):
    return 'S%02d' % i


def parse_program(prog, n):
    child = [-1] * n
    for tok in prog.split(' '):
        ps, c = tok.split('>')
        for p in ps.split('+'):
            child[int(p[1:])] = int(c[1:])
    return np.array(child)


def load_networks(path):
    df = pd.read_csv(path)
    nets = []
    for nid, sub in df.groupby('network_id', sort=True):
        n = int(sub.channel_count.iloc[0])
        X = np.zeros((len(sub), n, 4, 96), np.float32)
        RK = np.zeros((len(sub), n, 2), np.float32)
        for w, s in enumerate(sub.signals_json):
            for ch in json.loads(s):
                i = int(ch['slot'][1:])
                RK[w, i, 0] = ch['flow_scale_rank']
                RK[w, i, 1] = ch['flow_variability_rank']
                for k, f in enumerate(SIGS):
                    X[w, i, k] = ch[f]
        d = dict(nid=nid, n=n, X=X, RK=RK, ids=list(sub['id']))
        if 'relay_program' in sub:
            d['child'] = parse_program(sub.relay_program.iloc[0], n)
        nets.append(d)
    return nets


def window_subset(d, widx):
    dd = dict(d)
    dd['X'] = d['X'][widx]
    dd['RK'] = d['RK'][widx]
    return dd


# ----------------------------------------------------------------------------- base pairwise features
def _std(a):
    m = a.mean(-1, keepdims=True)
    s = a.std(-1, keepdims=True)
    return (a - m) / np.where(s > 1e-6, s, 1.0), (s[..., 0] > 1e-6)


def lagcorr(A, lags):
    """A: (W, n, T) -> (W, len(lags), n, n) with corr(u[t], v[t+L])."""
    W, n, T = A.shape
    out = np.zeros((W, len(lags), n, n), np.float32)
    for k, L in enumerate(lags):
        a, b = (A[:, :, :T - L], A[:, :, L:]) if L >= 0 else (A[:, :, -L:], A[:, :, :T + L])
        za, va = _std(a)
        zb, vb = _std(b)
        out[:, k] = np.einsum('wit,wjt->wij', za, zb) / a.shape[-1] * va[:, :, None] * vb[:, None, :]
    return out


def node_feats(d):
    X, RK, n = d['X'], d['RK'], d['n']
    sr = RK[:, :, 0] / (n - 1)
    ndl = X[:, :, 1]
    f = {'sr_mean': sr.mean(0), 'sr_std': sr.std(0), 'sr_top_frac': (RK[:, :, 0] == n - 1).mean(0),
         'vr_mean': (RK[:, :, 1] / (n - 1)).mean(0), 'ndl_mean': ndl.mean((0, 2)), 'ndl_std': ndl.std(2).mean(0),
         'ndl_max': ndl.max(2).mean(0), 'rain_mean': X[:, :, 3].mean((0, 2)), 'dl_std': X[:, :, 0].std(2).mean(0),
         'dc_abs': np.abs(X[:, :, 2]).mean((0, 2))}
    f['ndl_rank'] = np.argsort(np.argsort(f['ndl_mean'])) / (n - 1)
    f['sr_rank'] = np.argsort(np.argsort(f['sr_mean'])) / (n - 1)
    return f


def base_pair_feats(d):
    X, RK, n = d['X'], d['RK'], d['n']
    nf = node_feats(d)
    i0 = LAGS.index(0)
    ccs = {'dl': lagcorr(X[:, :, 0], LAGS).mean(0), 'dc': lagcorr(X[:, :, 2], LAGS).mean(0),
           'ndl': lagcorr(X[:, :, 1], LAGS).mean(0)}
    feats = {}
    sr = RK[:, :, 0]
    feats['d_sr'] = (sr[:, None, :] - sr[:, :, None]).mean(0) / (n - 1)
    feats['frac_v_higher'] = (sr[:, None, :] > sr[:, :, None]).mean(0)
    feats['d_vr'] = (RK[:, None, :, 1] - RK[:, :, None, 1]).mean(0) / (n - 1)
    ndl = X[:, :, 1]
    dn = ndl[:, None, :, :] - ndl[:, :, None, :]
    feats['d_ndl_mean'] = dn.mean((0, 3))
    feats['d_ndl_frac_pos'] = (dn > 0).mean((0, 3))
    feats['d_ndl_frac_neg'] = (dn < 0).mean((0, 3))
    for name, cc in ccs.items():
        feats['cc_%s_0' % name] = cc[i0]
        feats['cc_%s_max' % name] = cc.max(0)
        feats['cc_%s_argmax' % name] = np.array(LAGS)[cc.argmax(0)]
        feats['cc_%s_pos' % name] = cc[i0 + 1:].max(0)
        feats['cc_%s_neg' % name] = cc[:i0].max(0)
        feats['cc_%s_asym' % name] = feats['cc_%s_pos' % name] - feats['cc_%s_neg' % name]
        for L in (1, 2, 3, 4, 6):
            feats['cc_%s_l%d' % (name, L)] = cc[i0 + L]
    feats['rain_cc'] = lagcorr(X[:, :, 3], [0]).mean(0)[0]
    eye = np.eye(n, dtype=bool)
    for key in ['cc_dl_max', 'cc_dc_max', 'cc_dc_pos', 'rain_cc', 'cc_dl_0']:
        M = feats[key].copy()
        M[eye] = -9
        feats[key + '_gap'] = M - M.max(1, keepdims=True)
        feats[key + '_rk'] = (M[:, None, :] > M[:, :, None]).sum(2)
        Mh = np.where(feats['d_sr'] > 0, M, -9)
        feats[key + '_hgap'] = Mh - Mh.max(1, keepdims=True)
        feats[key + '_hrk'] = (Mh[:, None, :] > Mh[:, :, None]).sum(2)
    D = feats['d_sr']
    Dh = np.where(D > 0, D, 9)
    Dh[eye] = 9
    feats['dsr_hrk'] = (Dh[:, None, :] < Dh[:, :, None]).sum(2)
    feats['n_higher'] = np.repeat((Dh < 9).sum(1, keepdims=True), n, 1)
    C = feats['cc_dl_max']
    between = np.full((n, n), -1.0, np.float32)
    for u in range(n):
        for v in range(n):
            for w in range(n):
                if w != u and w != v and D[u, w] > 0 and D[w, v] > 0:
                    between[u, v] = max(between[u, v], min(C[u, w], C[w, v]))
    feats['between_cc'] = between
    feats['between_gap'] = C - between
    for k, a in nf.items():
        feats['u_' + k] = np.repeat(a[:, None], n, 1)
        feats['v_' + k] = np.repeat(a[None, :], n, 0)
    feats['n'] = np.full((n, n), n, np.float32)
    return feats


# ----------------------------------------------------------------------------- extra pairwise evidence
def _z(A):
    Z = A - A.mean(-1, keepdims=True)
    s = Z.std(-1, keepdims=True)
    return Z / np.where(s > 1e-6, s, 1.0), (s[..., 0] > 1e-6)


def corrW(A):
    Z, ok = _z(A)
    C = np.einsum('wit,wjt->wij', Z, Z) / A.shape[-1]
    return C * ok[:, :, None] * ok[:, None, :]


def rel_feats(M, key, D, eye, out):
    """candidate-relative versions of M for each u over v (all candidates and higher-scale only)."""
    M = M.astype(np.float64).copy()
    M[eye] = -9
    out[key + '_gap'] = M - M.max(1, keepdims=True)
    srt = np.sort(M, 1)
    second = srt[:, -2:-1]
    out[key + '_gap2'] = M - np.where(M >= srt[:, -1:], second, srt[:, -1:])
    out[key + '_rk'] = (M[:, None, :] > M[:, :, None]).sum(2)
    Mh = np.where(D > 0, M, -9)
    Mh[eye] = -9
    out[key + '_hgap'] = Mh - Mh.max(1, keepdims=True)
    out[key + '_hrk'] = (Mh[:, None, :] > Mh[:, :, None]).sum(2)
    out[key + '_cgap'] = M - M.max(0, keepdims=True)


def ema(x, a):
    y = np.zeros_like(x)
    acc = x[..., 0].copy()
    for t in range(x.shape[-1]):
        acc = a * x[..., t] + (1 - a) * acc
        y[..., t] = acc
    return y


def group_events(d, feats):
    """rising-limb onsets of u followed by onsets of v within a lag band."""
    dc = d['X'][:, :, 2]
    T = dc.shape[-1]
    on = np.zeros_like(dc, dtype=bool)
    on[:, :, 1:] = (dc[:, :, 1:] > 0) & (dc[:, :, :-1] <= 0)
    strong = np.zeros_like(dc, dtype=bool)
    strong[:, :, 1:] = (dc[:, :, 1:] >= 2) & (dc[:, :, :-1] < 2)
    for nm, E in [('on', on), ('st', strong)]:
        cnt_u = E.sum(2).astype(np.float64)
        for k0, k1, tag in [(0, 0, '0'), (0, 2, '02'), (1, 3, '13'), (-3, -1, 'm31')]:
            Vs = np.zeros_like(E)
            for L in range(k0, k1 + 1):
                if L >= 0:
                    Vs[:, :, :T - L] |= E[:, :, L:]
                else:
                    Vs[:, :, -L:] |= E[:, :, :T + L]
            hit = np.einsum('wit,wjt->wij', E.astype(np.float32), Vs.astype(np.float32))
            feats['ev_%s_%s' % (nm, tag)] = hit.sum(0) / np.maximum(cnt_u.sum(0)[:, None], 1)
        feats['ev_%s_rate_ratio' % nm] = np.log((E.sum((0, 2))[None, :] + 1) / (E.sum((0, 2))[:, None] + 1))


def group_rain(d, feats):
    X, n = d['X'], d['n']
    R = X[:, :, 3]
    feats['rain_mad'] = -np.abs(R[:, :, None] - R[:, None]).mean((0, 3))
    feats['rain_eq'] = (R[:, :, None] == R[:, None]).mean((0, 3))
    feats['rain_dcc'] = corrW(np.diff(R, axis=-1)).mean(0)
    feats['rain_rcc'] = corrW(R - R.mean(1, keepdims=True)).mean(0)
    lc = lagcorr(R, [-2, -1, 1, 2]).mean(0)
    feats['rain_lag_asym1'] = lc[2] - lc[1]
    feats['rain_lag_asym2'] = lc[3] - lc[0]
    feats['rain_dmean'] = (R[:, None, :] - R[:, :, None]).mean((0, 3))
    resp = []
    for a in (0.5, 0.2, 0.05):
        Rs = ema(R, a)
        c = np.array([[np.corrcoef(Rs[w, i], X[w, i, 0])[0, 1] if Rs[w, i].std() > 0 and X[w, i, 0].std() > 0 else 0
                       for i in range(n)] for w in range(R.shape[0])]).mean(0)
        resp.append(c)
    resp = np.array(resp)
    best_a = resp.argmax(0).astype(np.float64)
    feats['rresp_d_besta'] = best_a[None, :] - best_a[:, None]
    feats['rresp_d_fast'] = resp[0][None, :] - resp[0][:, None]
    feats['rresp_d_slow'] = resp[2][None, :] - resp[2][:, None]
    Rs = ema(R, 0.2)
    Zr, _ = _z(Rs)
    Zd, _ = _z(X[:, :, 0])
    C = np.einsum('wit,wjt->wij', Zr, Zd).mean(0) / R.shape[-1]
    own = np.diag(C)
    feats['xresp_u2v'] = C - own[None, :]
    feats['xresp_v2u'] = C.T - own[:, None]


def group_nnls(d, feats, srm, lags=(0, 1), rain_alphas=(0.5, 0.15), sig=0, tag='dl'):
    """per window NNLS: centred flow_v(t) ~ sum of lagged lower-scale candidate flows + own-rain terms.
    Features: coefficient mass of u, fit loss when u is removed, share of explained signal, single-u gain."""
    X, n = d['X'], d['n']
    W, T = X.shape[0], X.shape[-1]
    S = X[:, :, sig].astype(np.float64)
    R = X[:, :, 3].astype(np.float64)
    Lm = max(lags)
    coef = np.full((n, n), np.nan)
    drop = np.full((n, n), np.nan)
    share = np.full((n, n), np.nan)
    single = np.full((n, n), np.nan)
    r2 = np.full(n, np.nan)
    r2rain = np.full(n, np.nan)
    rain_regs = [ema(R, a) for a in rain_alphas]
    for v in range(n):
        cand = [w for w in range(n) if w != v and srm[w] < srm[v]]
        if not cand:
            continue
        cs = np.zeros((W, len(cand)))
        dr = np.zeros((W, len(cand)))
        sh = np.zeros((W, len(cand)))
        sg = np.zeros((W, len(cand)))
        r2s = np.zeros(W)
        r2r = np.zeros(W)
        for w_ in range(W):
            y = S[w_, v, Lm:]
            y = y - y.mean()
            sst = (y ** 2).sum() + 1e-9
            cols, owner = [], []
            for j, c in enumerate(cand):
                for L in lags:
                    x = S[w_, c, Lm - L:T - L]
                    cols.append(x - x.mean()); owner.append(j)
            for rr in rain_regs:
                x = rr[w_, v, Lm:]
                cols.append(x - x.mean()); owner.append(-1)
                cols.append(-(x - x.mean())); owner.append(-1)
            A = np.array(cols).T
            owner = np.array(owner)
            b, res = nnls(A, y)
            sse = res ** 2
            r2s[w_] = 1 - sse / sst
            Ar = A[:, owner == -1]
            _, resr = nnls(Ar, y)
            r2r[w_] = 1 - resr ** 2 / sst
            tot = np.abs(A * b[None]).sum() + 1e-9
            for j in range(len(cand)):
                m = owner == j
                cs[w_, j] = b[m].sum()
                sh[w_, j] = np.abs(A[:, m] * b[m][None]).sum() / tot
                _, resj = nnls(A[:, ~m], y)
                dr[w_, j] = (resj ** 2 - sse) / sst
                _, ress = nnls(np.concatenate([A[:, m], Ar], 1), y)
                sg[w_, j] = (resr ** 2 - ress ** 2) / sst
        for j, c in enumerate(cand):
            coef[c, v] = cs[:, j].mean()
            drop[c, v] = dr[:, j].mean()
            share[c, v] = sh[:, j].mean()
            single[c, v] = sg[:, j].mean()
        r2[v] = r2s.mean()
        r2rain[v] = r2r.mean()
    feats['nn%s_coef' % tag] = coef
    feats['nn%s_drop' % tag] = drop
    feats['nn%s_share' % tag] = share
    feats['nn%s_single' % tag] = single
    feats['nn%s_v_r2' % tag] = np.repeat(r2[None, :], n, 0)
    feats['nn%s_v_r2rain' % tag] = np.repeat(r2rain[None, :], n, 0)


REL_KEYS = ('rain_mad', 'rain_eq', 'rain_dcc', 'rain_rcc', 'pc_dl_btw', 'pc_dl_min', 'pc_dc_btw', 'nndl_coef',
            'nndl_drop', 'nndl_share', 'nndl_single', 'ev_on_02', 'ev_st_02', 'xresp_u2v', 'nndc_coef', 'nndc_drop')


def pair_feats(d):
    feats = base_pair_feats(d)
    n = d['n']
    D = feats['d_sr']
    eye = np.eye(n, dtype=bool)
    srm = d['RK'][:, :, 0].mean(0)
    new = {}
    group_events(d, new)
    group_rain(d, new)
    group_nnls(d, new, srm)
    relk = [k for k in new if (k.startswith('nn') and k.split('_')[-1] in ('coef', 'drop', 'share', 'single'))
            or k in REL_KEYS]
    for k in relk:
        M = np.where(np.isnan(new[k]), -9, new[k])
        out = {}
        rel_feats(M, k, D, eye, out)
        for kk, vv in out.items():
            vv = vv.astype(np.float64)
            vv[np.isnan(new[k])] = np.nan
            new[kk] = vv
    feats.update(new)
    names = sorted(feats)
    return np.stack([np.asarray(feats[k], np.float64) for k in names], -1).astype(np.float32)


def root_feats(d):
    nf = node_feats(d)
    F = np.stack([nf[k] for k in sorted(nf)], -1)
    return np.concatenate([F, np.full((d['n'], 1), d['n'])], 1).astype(np.float32)


def candidate_rows(d):
    """(n*n, k) rows in u-major order; row (u, v) scores 'child of u is v', row (u, u) scores 'u is outlet'."""
    F, R, n = pair_feats(d), root_feats(d), d['n']
    k, m = F.shape[-1], R.shape[1]
    X = np.full((n, n, k + 1 + m), np.nan, np.float32)
    X[:, :, :k] = F
    X[:, :, k] = 0.0
    ii = np.arange(n)
    X[ii, ii, :k] = np.nan
    X[ii, ii, k] = 1.0
    X[ii, ii, k + 1:] = R
    Y = None
    if 'child' in d:
        Y = np.zeros((n, n), np.float32)
        for u in range(n):
            Y[u, u if d['child'][u] < 0 else d['child'][u]] = 1
        Y = Y.reshape(-1)
    return X.reshape(n * n, -1), Y


# ----------------------------------------------------------------------------- model
def softmax_objective(gid):
    """per-node softmax cross-entropy over a node's candidate rows (gid = node group of each row)."""
    ng = gid.max() + 1

    def fobj(preds, data):
        y = data.get_label()
        mx = np.full(ng, -np.inf)
        np.maximum.at(mx, gid, preds)
        e = np.exp(preds - mx[gid])
        s = np.zeros(ng)
        np.add.at(s, gid, e)
        p = e / s[gid]
        return p - y, np.maximum(p * (1 - p), 1e-6)
    return fobj


def training_set(nets):
    rng = np.random.RandomState(SEED)
    Xs, Ys, G = [], [], []
    g = 0
    for d in nets:
        Wn = d['X'].shape[0]
        for _ in range(SUB_DRAWS):
            widx = np.sort(rng.choice(Wn, min(SUB_K, Wn), replace=False))
            X, Y = candidate_rows(window_subset(d, widx))
            n = d['n']
            Xs.append(X); Ys.append(Y)
            G.append(np.repeat(np.arange(g, g + n), n))
            g += n
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(G)


def train_model(X, Y, gid):
    p = dict(PARAMS)
    p['objective'] = softmax_objective(gid)
    return lgb.train(p, lgb.Dataset(X, Y), num_boost_round=ROUNDS)


def head_logprobs(raw, n):
    S = raw.reshape(n, n).astype(np.float64)
    S = S - S.max(1, keepdims=True)
    return S - np.log(np.exp(S).sum(1, keepdims=True))


# ----------------------------------------------------------------------------- decoding
_TREES = {}


def enumerate_trees(n):
    """All rooted labelled trees on n nodes as child arrays (-1 = outlet); n^(n-1) rows."""
    if n not in _TREES:
        F = np.indices([n] * n, dtype=np.int8).reshape(n, -1).T
        ar = np.arange(n, dtype=np.int8)
        F = F[(F == ar[None, :]).sum(1) == 1]
        g = F.copy()
        steps = 1
        while steps < n:
            g = np.take_along_axis(g, g.astype(np.int64), axis=1)
            steps *= 2
        T = F[(g == g[:, :1]).all(1)].copy()
        T[T == ar[None, :]] = -1
        assert len(T) == n ** (n - 1)
        _TREES[n] = T
    return _TREES[n]


def decode_tree(n, S):
    T = enumerate_trees(n)
    idx = np.where(T < 0, np.arange(n)[None, :], T).astype(np.int64)
    return T[np.argmax(S[np.arange(n)[None, :], idx].sum(1))].astype(int)


def to_program(child):
    parents = {}
    for i, c in enumerate(child):
        if c >= 0:
            parents.setdefault(int(c), []).append(i)
    depth = {}

    def dep(c):
        if c not in depth:
            depth[c] = max([1] + [1 + dep(p) for p in parents[c] if p in parents])
        return depth[c]

    toks = sorted((dep(c), '+'.join(slot_name(p) for p in sorted(ps)) + '>' + slot_name(c))
                  for c, ps in parents.items())
    return ' '.join(t for _, t in toks)


# ----------------------------------------------------------------------------- main
def main(public_dir, out_path):
    np.random.seed(SEED)
    tr = load_networks(os.path.join(public_dir, 'train.csv'))
    te = load_networks(os.path.join(public_dir, 'test.csv'))
    X, Y, gid = training_set(tr)
    print('train rows', X.shape, flush=True)
    model = train_model(X, Y, gid)
    prog = {}
    for d in te:
        Xt, _ = candidate_rows(d)
        S = head_logprobs(model.predict(Xt, raw_score=True), d['n'])
        p = to_program(decode_tree(d['n'], S))
        for rid in d['ids']:
            prog[rid] = p
    order = pd.read_csv(os.path.join(public_dir, 'test.csv'), usecols=['id'])['id']
    sub = pd.DataFrame({'id': order, 'relay_program': [prog[i] for i in order]})
    assert sub['relay_program'].notna().all() and sub['id'].is_unique
    sub.to_csv(out_path, index=False)
    print('wrote', out_path, len(sub), 'rows', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
