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
- Second learned family (neural_logprobs): four from-scratch permutation-equivariant neural networks
  (NetV1, Net2; channel CNN encoders over the raw series + pair modules over lagged cross-correlation
  tensors, per-node softmax over candidate children / outlet), trained in this script on all training
  networks (NN_COMPONENTS x NN_SEEDS fixed seeds, nothing pretrained). Their mean head log-probabilities are combined
  log-linearly with the LightGBM head log-probabilities (weight NN_WEIGHT) before tree decoding.
- Bagging: GBM_BAG LightGBM models, each with its own fixed seed and its own random window-subset draw;
  their per-node log-probabilities are averaged.
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
- CPU only. Measured runtime: about 11 minutes on 4 CPU cores.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
import torch
import torch.nn as nn
import torch.nn.functional as F

THREADS = 4
SEED = 0
ROUNDS = 1000
SUB_K = 5
SUB_DRAWS = 4
LAGS = list(range(-6, 13))
SIGS = ['discharge_level', 'network_discharge_level', 'discharge_change', 'rainfall_level']
NN_WEIGHT = 0.5
GBM_BAG = 3
NN_SEEDS = 2
NN_COMPONENTS = [
    ('v1', 'v1', dict(epochs=40)),
    ('pool1', 'n2', dict(epochs=30, net=dict(mode='pool'), reps=2, gb=4)),
    ('pool2ex', 'n2', dict(epochs=30, net=dict(mode='pool'), reps=2, gb=4, extra=1)),
    ('win2ex', 'n2', dict(epochs=30, net=dict(mode='window'), extra=1)),
]
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


def training_set(nets, seed=SEED):
    rng = np.random.RandomState(seed)
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


def train_model(X, Y, gid, seed=SEED):
    p = dict(PARAMS)
    p['seed'] = seed
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


# ----------------------------------------------------------------------------- neural family (from scratch)
NLAGS = list(range(-8, 17))
# (signal of u, signal of v)
KPAIRS = [(0, 0), (1, 1), (2, 2), (3, 3), (3, 0), (0, 3)]


def _nz(a):
    m = a.mean(-1, keepdims=True)
    s = a.std(-1, keepdims=True)
    ok = s > 1e-6
    return (a - m) / np.where(ok, s, 1.0), ok[..., 0]


def lagcorr2(A, B, lags):
    """A,B: (W, n, T). returns (W, n, n, len(lags)) corr(A_u[t], B_v[t+L])."""
    W, n, T = A.shape
    out = np.zeros((W, n, n, len(lags)), np.float32)
    for k, L in enumerate(lags):
        if L >= 0:
            a = A[:, :, :T - L]; b = B[:, :, L:]
        else:
            a = A[:, :, -L:]; b = B[:, :, :T + L]
        za, va = _nz(a)
        zb, vb = _nz(b)
        c = np.einsum('wit,wjt->wij', za, zb) / a.shape[-1]
        out[:, :, :, k] = c * va[:, :, None] * vb[:, None, :]
    return out


def prep_net(d):
    X = d['X'].astype(np.float32)
    W, n = X.shape[:2]
    CC = np.stack([lagcorr2(X[:, :, a], X[:, :, b], NLAGS) for a, b in KPAIRS], 3)  # (W,n,n,K,L)
    ndl = X[:, :, 1]
    dn = ndl[:, None, :, :] - ndl[:, :, None, :]  # (W,u,v,T) v-u
    dl = X[:, :, 0]
    ddl = dl[:, None] - dl[:, :, None]
    rk = d['RK'] / (n - 1)
    pf = [dn.mean(-1) / 5.0, (dn > 0).mean(-1), (dn < 0).mean(-1), np.abs(dn).mean(-1) / 5.0,
          np.abs(ddl).mean(-1) / 3.0,
          rk[:, None, :, 0] - rk[:, :, None, 0], rk[:, None, :, 1] - rk[:, :, None, 1],
          (rk[:, None, :, 0] > rk[:, :, None, 0]).astype(np.float32)]
    PF = np.stack(pf, -1).astype(np.float32)
    Xs = X.copy()
    Xs[:, :, 0] /= 4.0; Xs[:, :, 1] /= 5.0; Xs[:, :, 2] /= 2.0; Xs[:, :, 3] = Xs[:, :, 3] / 4.5 - 1.0
    return dict(X=Xs, RK=rk.astype(np.float32), CC=CC.astype(np.float32), PF=PF, n=n,
                child=d.get('child'))



class NetV1(nn.Module):
    def __init__(self, d=64, c=32, K=6, L=25, P=8, natt=1, ntri=1, rowctx=True, drop=0.1, extra=0, cc_mode='conv'):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(6, c, 7, padding=3), nn.GELU(),
            nn.Conv1d(c, c, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(c, c, 5, padding=8, dilation=4), nn.GELU())
        self.henc = nn.Sequential(nn.Linear(2 * c + 2, d), nn.GELU(), nn.Linear(d, d))
        self.att = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=drop, batch_first=True,
                                                             norm_first=True) for _ in range(natt)])
        self.cc_mode = cc_mode
        if cc_mode == 'conv':
            self.ccnet = nn.Sequential(nn.Conv1d(K, 16, 5, padding=2), nn.GELU(), nn.Flatten(), nn.Linear(16 * L, d))
        else:
            self.ccnet = nn.Sequential(nn.Flatten(), nn.Linear(K * L, d))
        self.pfl = nn.Linear(P + extra, d)
        self.hu = nn.Linear(d, d)
        self.hv = nn.Linear(d, d)
        self.huv = nn.Linear(d, d)
        self.pmlp = nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ntri = ntri
        self.tri = nn.ModuleList()
        for _ in range(ntri):
            self.tri.append(nn.ModuleDict(dict(
                ln=nn.LayerNorm(d), a=nn.Linear(d, d), b=nn.Linear(d, d), ga=nn.Linear(d, d),
                a2=nn.Linear(d, d), b2=nn.Linear(d, d), o=nn.Linear(2 * d, d), lno=nn.LayerNorm(2 * d),
                rmean=nn.Linear(d, d), rmax=nn.Linear(d, d),
                ff=nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)))))
        self.rowctx = rowctx
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.root = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, X, RK, CC, PF):
        # X (B,n,4,T) RK (B,n,2) CC (B,n,n,K,L) PF (B,n,n,P)
        B, n, _, T = X.shape
        xin = torch.cat([X, RK[..., None].expand(B, n, 2, T)], 2).reshape(B * n, 6, T)
        z = self.enc(xin)
        h = torch.cat([z.mean(-1), z.amax(-1)], -1).reshape(B, n, -1)
        h = self.henc(torch.cat([h, RK], -1))
        for a in self.att:
            h = a(h)
        K, L = CC.shape[-2:]
        ce = self.ccnet(CC.reshape(B * n * n, K, L)).reshape(B, n, n, -1)
        e = ce + self.pfl(PF) + self.hu(h)[:, :, None] + self.hv(h)[:, None, :] + self.huv(h[:, :, None] * h[:, None, :])
        e = self.pmlp(e)
        eye = torch.eye(n, dtype=torch.bool, device=X.device)
        offd = (~eye).float()[None, :, :, None]
        for t in self.tri:
            x = t['ln'](e)
            g = torch.sigmoid(t['ga'](x))
            a = t['a'](x) * offd
            b = t['b'](x) * offd
            # path u->w->v : sum_w a[u,w] * b[w,v]
            p1 = torch.einsum('buwd,bwvd->buvd', a, b) / n
            a2 = t['a2'](x) * offd
            b2 = t['b2'](x) * offd
            # shared target: sum_w a2[u,w]*b2[v,w]  (u and v both near w)
            p2 = torch.einsum('buwd,bvwd->buvd', a2, b2) / n
            e = e + g * t['o'](t['lno'](torch.cat([p1, p2], -1)))
            if self.rowctx:
                xm = (x * offd).sum(2, keepdim=True) / (n - 1)
                xx = (x - 1e4 * (1 - offd)).amax(2, keepdim=True)
                e = e + t['rmean'](xm) + t['rmax'](xx)
            e = e + t['ff'](e)
        s = self.out(e)[..., 0]  # (B,n,n)
        hg = torch.cat([h, h.mean(1, keepdim=True).expand(B, n, -1), h.amax(1, keepdim=True).expand(B, n, -1)], -1)
        r = self.root(hg)[..., 0]
        s = torch.where(eye[None], r[:, :, None].expand(B, n, n), s)
        return s


def v1_net_tensors(p, extra=None):
    X = torch.from_numpy(p['X'])
    RK = torch.from_numpy(p['RK'])
    CC = torch.from_numpy(p['CC'])
    PF = torch.from_numpy(p['PF'])
    if extra is not None:
        PF = torch.cat([PF, torch.from_numpy(extra).float()], -1)
    return X, RK, CC, PF


def v1_targets(p):
    ch = p['child']
    n = p['n']
    return torch.from_numpy(np.where(ch < 0, np.arange(n), ch).astype(np.int64))


def v1_predict_net(model, p, extra=None, bs=64):
    model.eval()
    X, RK, CC, PF = v1_net_tensors(p, extra)
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            s = model(X[i:i + bs], RK[i:i + bs], CC[i:i + bs], PF[i:i + bs])
            out.append(F.log_softmax(s, -1))
    lp = torch.cat(out).mean(0).numpy().astype(np.float64)
    lp = lp - lp.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    return lp


def v1_train_model(P, idx, cfg, extras=None, eval_fn=None, seed=0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = NetV1(**cfg.get('net', {}))
    epochs = cfg.get('epochs', 30)
    bs = cfg.get('bs', 32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 2e-3), weight_decay=cfg.get('wd', 1e-4))
    # group windows by n
    groups = {}
    for i in idx:
        p = P[i]
        groups.setdefault(p['n'], []).append(i)
    data = {}
    for n, ids in groups.items():
        Xs, RKs, CCs, PFs, Ys = [], [], [], [], []
        for i in ids:
            X, RK, CC, PF = v1_net_tensors(P[i], None if extras is None else extras[i])
            W = X.shape[0]
            Xs.append(X); RKs.append(RK); CCs.append(CC); PFs.append(PF); Ys.append(v1_targets(P[i])[None].expand(W, n))
        data[n] = [torch.cat(a) for a in (Xs, RKs, CCs, PFs, Ys)]
    nb_per_epoch = sum(int(math.ceil(len(v[0]) / bs)) for v in data.values())
    total = epochs * nb_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get('lr', 2e-3), total_steps=total,
                                                pct_start=cfg.get('warm', 0.1))
    hist = []
    for ep in range(epochs):
        model.train()
        batches = []
        for n, v in data.items():
            perm = rng.permutation(len(v[0]))
            for j in range(0, len(perm), bs):
                batches.append((n, perm[j:j + bs]))
        order = rng.permutation(len(batches))
        tl = 0.0
        for bi in order:
            n, b = batches[bi]
            X, RK, CC, PF, Y = [a[b] for a in data[n]]
            if cfg.get('noise', 0) > 0:
                X = X + cfg['noise'] * torch.randn_like(X)
                CC = CC + cfg['noise'] * torch.randn_like(CC)
            s = model(X, RK, CC, PF)
            loss = F.cross_entropy(s.reshape(-1, n), Y.reshape(-1), label_smoothing=cfg.get('ls', 0.0))
            opt.zero_grad()
            loss.backward()
            if cfg.get('clip', 1.0):
                nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 1.0))
            opt.step()
            sched.step()
            tl += loss.item()
        if eval_fn is not None and ((ep + 1) in cfg.get('eval_at', [epochs])):
            hist.append((ep + 1, eval_fn(model)))
        if verbose:
            print('ep %d loss %.4f' % (ep + 1, tl / len(order)), flush=True)
    return model, hist



class Tri2(nn.Module):
    def __init__(self, d, rowctx=True):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.a, self.b, self.ga = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.a2, self.b2 = nn.Linear(d, d), nn.Linear(d, d)
        self.o, self.lno = nn.Linear(2 * d, d), nn.LayerNorm(2 * d)
        self.rmean, self.rmax = nn.Linear(d, d), nn.Linear(d, d)
        self.rowctx = rowctx
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, e, offd):
        n = e.shape[1]
        x = self.ln(e)
        g = torch.sigmoid(self.ga(x))
        a = self.a(x) * offd
        b = self.b(x) * offd
        p1 = torch.einsum('buwd,bwvd->buvd', a, b) / n
        a2 = self.a2(x) * offd
        b2 = self.b2(x) * offd
        p2 = torch.einsum('buwd,bvwd->buvd', a2, b2) / n
        e = e + g * self.o(self.lno(torch.cat([p1, p2], -1)))
        if self.rowctx:
            xm = (x * offd).sum(2, keepdim=True) / (n - 1)
            xx = (x - 1e4 * (1 - offd)).amax(2, keepdim=True)
            e = e + self.rmean(xm) + self.rmax(xx)
        return e + self.ff(e)


class Net2(nn.Module):
    def __init__(self, d=64, c=32, K=6, L=25, P=8, natt=1, ntri=1, ntri_w=0, rowctx=True, drop=0.1, extra=0,
                 mode='window', nodeatt=1, xcorr=0, aux=0):
        super().__init__()
        self.mode = mode
        self.enc = nn.Sequential(
            nn.Conv1d(6, c, 7, padding=3), nn.GELU(),
            nn.Conv1d(c, c, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(c, c, 5, padding=8, dilation=4), nn.GELU())
        self.xcorr = xcorr
        if xcorr:
            # learned cross-correlation: project feature maps to xcorr channels, correlate at lags
            self.xp = nn.Conv1d(c, xcorr, 1)
            self.xnet = nn.Sequential(nn.Conv1d(xcorr, 16, 5, padding=2), nn.GELU(), nn.Flatten(), nn.Linear(16 * L, d))
        self.henc = nn.Sequential(nn.Linear(2 * c + 2, d), nn.GELU(), nn.Linear(d, d))
        self.att = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=drop, batch_first=True,
                                                             norm_first=True) for _ in range(natt)])
        self.ccnet = nn.Sequential(nn.Conv1d(K, 16, 5, padding=2), nn.GELU(), nn.Flatten(), nn.Linear(16 * L, d))
        self.pfl = nn.Linear(P, d)
        self.extra = extra
        if extra:
            self.exl = nn.Sequential(nn.Linear(extra, d), nn.GELU(), nn.Linear(d, d))
        self.hu, self.hv, self.huv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.pmlp = nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.tri_w = nn.ModuleList([Tri2(d, rowctx) for _ in range(ntri_w)])
        if mode == 'pool':
            self.pool_e = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
            self.pool_h = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
            self.natt2 = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=drop, batch_first=True,
                                                                   norm_first=True) for _ in range(nodeatt)])
            self.hu2, self.hv2 = nn.Linear(d, d), nn.Linear(d, d)
        self.tri = nn.ModuleList([Tri2(d, rowctx) for _ in range(ntri)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.root = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
        self.aux = aux
        if aux:
            self.auxl = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self._aux = None

    def lagx(self, z):
        # z (B,n,C,T) -> (B,n,n,C,L) normalized cross-corr at lags -8..16 (corr(z_u[t], z_v[t+L]))
        B, n, C, T = z.shape
        outs = []
        for Lg in NLAGS:
            if Lg >= 0:
                a = z[..., :T - Lg]; b = z[..., Lg:]
            else:
                a = z[..., -Lg:]; b = z[..., :T + Lg]
            a = (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + 1e-3)
            b = (b - b.mean(-1, keepdim=True)) / (b.std(-1, keepdim=True) + 1e-3)
            outs.append(torch.einsum('bict,bjct->bijc', a, b) / a.shape[-1])
        return torch.stack(outs, -1)

    def encode(self, X, RK, CC, PF, EX=None):
        B, n, _, T = X.shape
        xin = torch.cat([X, RK[..., None].expand(B, n, 2, T)], 2).reshape(B * n, 6, T)
        z = self.enc(xin)
        h = torch.cat([z.mean(-1), z.amax(-1)], -1).reshape(B, n, -1)
        h = self.henc(torch.cat([h, RK], -1))
        for a in self.att:
            h = a(h)
        K, L = CC.shape[-2:]
        ce = self.ccnet(CC.reshape(B * n * n, K, L)).reshape(B, n, n, -1)
        e = ce + self.pfl(PF) + self.hu(h)[:, :, None] + self.hv(h)[:, None, :] + self.huv(h[:, :, None] * h[:, None, :])
        if self.xcorr:
            zx = self.xp(z).reshape(B, n, self.xcorr, T)
            xc = self.lagx(zx)
            e = e + self.xnet(xc.reshape(B * n * n, self.xcorr, L)).reshape(B, n, n, -1)
        if self.extra and EX is not None:
            e = e + self.exl(EX)
        e = self.pmlp(e)
        return h, e

    def head(self, h, e):
        B, n = h.shape[:2]
        eye = torch.eye(n, dtype=torch.bool)
        offd = (~eye).float()[None, :, :, None]
        for t in self.tri:
            e = t(e, offd)
        s = self.out(e)[..., 0]
        if self.aux:
            self._aux = self.auxl(e)[..., 0]
        hg = torch.cat([h, h.mean(1, keepdim=True).expand(B, n, -1), h.amax(1, keepdim=True).expand(B, n, -1)], -1)
        r = self.root(hg)[..., 0]
        return torch.where(eye[None], r[:, :, None].expand(B, n, n), s)

    def forward(self, X, RK, CC, PF, EX=None, G=None):
        """window mode: returns per-window scores (B,n,n). pool mode: X etc are (G*k, ...) with G networks of k windows
        each (contiguous), returns per-network scores (G,n,n)."""
        h, e = self.encode(X, RK, CC, PF, EX)
        n = h.shape[1]
        if self.mode == 'window':
            if self.tri_w:
                offd = (~torch.eye(n, dtype=torch.bool)).float()[None, :, :, None]
                for t in self.tri_w:
                    e = t(e, offd)
            return self.head(h, e)
        if self.tri_w:
            offd = (~torch.eye(n, dtype=torch.bool)).float()[None, :, :, None]
            for t in self.tri_w:
                e = t(e, offd)
        Bk = h.shape[0]
        k = Bk // G
        h = h.reshape(G, k, n, -1)
        e = e.reshape(G, k, n, n, -1)
        hp = self.pool_h(torch.cat([h.mean(1), h.std(1, unbiased=False) if k > 1 else torch.zeros_like(h[:, 0])], -1))
        ep = self.pool_e(torch.cat([e.mean(1), e.std(1, unbiased=False) if k > 1 else torch.zeros_like(e[:, 0])], -1))
        for a in self.natt2:
            hp = a(hp)
        ep = ep + self.hu2(hp)[:, :, None] + self.hv2(hp)[:, None, :]
        return self.head(hp, ep)


def n2_net_tensors(p, extra=None):
    X = torch.from_numpy(p['X'])
    RK = torch.from_numpy(p['RK'])
    CC = torch.from_numpy(p['CC'])
    PF = torch.from_numpy(p['PF'])
    EX = None if extra is None else torch.from_numpy(np.ascontiguousarray(extra)).float()
    return X, RK, CC, PF, EX


def n2_targets(p):
    ch = p['child']
    n = p['n']
    return torch.from_numpy(np.where(ch < 0, np.arange(n), ch).astype(np.int64))


def n2_anc_matrix(p):
    ch = p['child']
    n = p['n']
    A = np.zeros((n, n), np.float32)
    for u in range(n):
        j = ch[u]
        while j >= 0:
            A[u, j] = 1
            j = ch[j]
    return torch.from_numpy(A)


def n2_norm(lp):
    lp = lp - lp.max(1, keepdims=True)
    return lp - np.log(np.exp(lp).sum(1, keepdims=True))


def n2_predict_net(model, p, extra=None, bs=64):
    model.eval()
    X, RK, CC, PF, EX = n2_net_tensors(p, extra)
    with torch.no_grad():
        if model.mode == 'pool':
            s = model(X, RK, CC, PF, EX, G=1)[0]
            return n2_norm(F.log_softmax(s, -1).numpy().astype(np.float64))
        out = []
        for i in range(0, X.shape[0], bs):
            s = model(X[i:i + bs], RK[i:i + bs], CC[i:i + bs], PF[i:i + bs], None if EX is None else EX[i:i + bs])
            out.append(F.log_softmax(s, -1))
    return n2_norm(torch.cat(out).mean(0).numpy().astype(np.float64))


def n2_train_model(P, idx, cfg, extras=None, eval_fn=None, seed=0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    ncfg = dict(cfg.get('net', {}))
    model = Net2(**ncfg)
    mode = model.mode
    epochs = cfg.get('epochs', 30)
    bs = cfg.get('bs', 32)
    lr = cfg.get('lr', 2e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get('wd', 1e-4))
    groups = {}
    for i in idx:
        groups.setdefault(P[i]['n'], []).append(i)
    data = {}
    for n, ids in groups.items():
        cols = [[], [], [], [], [], [], []]
        for i in ids:
            t = n2_net_tensors(P[i], None if extras is None else extras[i])
            W = t[0].shape[0]
            for j in range(4):
                cols[j].append(t[j])
            cols[4].append(t[4] if t[4] is not None else torch.zeros(W, n, n, 0))
            cols[5].append(n2_targets(P[i])[None].expand(W, n))
            cols[6].append(n2_anc_matrix(P[i])[None].expand(W, n, n))
        data[n] = [torch.stack(a) for a in cols]  # (Nnet, W, ...)  W=10 for all train nets
    kmin, kmax = cfg.get('kmin', 3), cfg.get('kmax', 10)
    gb = cfg.get('gb', 4)  # networks per batch in pool mode
    reps = cfg.get('reps', 4)  # samples per network per epoch in pool mode

    def make_batches():
        bl = []
        for n, v in data.items():
            N, W = v[0].shape[:2]
            if mode == 'window':
                perm = rng.permutation(N * W)
                for j in range(0, len(perm), bs):
                    bl.append((n, perm[j:j + bs], None))
            else:
                for r in range(reps):
                    perm = rng.permutation(N)
                    for j in range(0, N, gb):
                        k = rng.randint(kmin, kmax + 1)
                        wsel = np.stack([rng.choice(W, k, replace=False) for _ in perm[j:j + gb]])
                        bl.append((n, perm[j:j + gb], wsel))
        return bl

    nb = len(make_batches())
    total = epochs * nb
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=cfg.get('warm', 0.1))
    hist = []
    useEX = extras is not None
    for ep in range(epochs):
        model.train()
        batches = make_batches()
        order = rng.permutation(len(batches))
        tl = 0.0
        for bi in order:
            n, b, wsel = batches[bi]
            v = data[n]
            if mode == 'window':
                N, W = v[0].shape[:2]
                ni, wi = b // W, b % W
                X, RK, CC, PF, EX, Y, A = [a[ni, wi] for a in v]
                G = None
            else:
                G, k = wsel.shape
                ni = np.repeat(b, k)
                wi = wsel.reshape(-1)
                X, RK, CC, PF, EX = [a[ni, wi] for a in v[:5]]
                Y = v[5][b, 0]
                A = v[6][b, 0]
            if cfg.get('crop'):
                lo, hi = cfg['crop']
                Lc = int(rng.randint(lo, hi + 1))
                o = int(rng.randint(0, 96 - Lc + 1))
                X = X[..., o:o + Lc].contiguous()
                CC, PF = n2_torch_feats(X, RK)
            if cfg.get('noise', 0) > 0:
                CC = CC + cfg['noise'] * torch.randn_like(CC)
            s = model(X, RK, CC, PF, EX if useEX else None, G=G)
            loss = F.cross_entropy(s.reshape(-1, n), Y.reshape(-1), label_smoothing=cfg.get('ls', 0.0))
            if model.aux:
                offm = ~torch.eye(n, dtype=torch.bool)[None].expand_as(A)
                loss = loss + model.aux * F.binary_cross_entropy_with_logits(model._aux[offm], A[offm])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 1.0))
            opt.step()
            sched.step()
            tl += loss.item()
        if eval_fn is not None and ((ep + 1) in cfg.get('eval_at', [epochs])):
            hist.append((ep + 1, eval_fn(model)))
        if verbose:
            print('ep %d loss %.4f' % (ep + 1, tl / len(order)), flush=True)
    return model, hist



def n2_torch_feats(X, RK):
    """Recompute CC (B,n,n,K,L) and PF (B,n,n,8) from scaled X (B,n,4,T) and RK (B,n,2); mirrors nprep."""
    B, n, _, T = X.shape
    cc = []
    for a_, b_ in KPAIRS:
        A = X[:, :, a_]; Bv = X[:, :, b_]
        lags = []
        for Lg in NLAGS:
            if Lg >= 0:
                a = A[..., :T - Lg]; b = Bv[..., Lg:]
            else:
                a = A[..., -Lg:]; b = Bv[..., :T + Lg]
            sa = a.std(-1, unbiased=False, keepdim=True); sb = b.std(-1, unbiased=False, keepdim=True)
            va = (sa > 1e-6).float(); vb = (sb > 1e-6).float()
            za = (a - a.mean(-1, keepdim=True)) / torch.where(sa > 1e-6, sa, torch.ones_like(sa))
            zb = (b - b.mean(-1, keepdim=True)) / torch.where(sb > 1e-6, sb, torch.ones_like(sb))
            c = torch.einsum('bit,bjt->bij', za, zb) / a.shape[-1]
            lags.append(c * va[:, :, None, 0] * vb[:, None, :, 0])
        cc.append(torch.stack(lags, -1))
    CC = torch.stack(cc, 3)
    ndl = X[:, :, 1]
    dn = ndl[:, None] - ndl[:, :, None]
    dl = X[:, :, 0]
    ddl = dl[:, None] - dl[:, :, None]
    rk = RK
    pf = [dn.mean(-1), (dn > 0).float().mean(-1), (dn < 0).float().mean(-1), dn.abs().mean(-1),
          ddl.abs().mean(-1) * 4.0 / 3.0,
          rk[:, None, :, 0] - rk[:, :, None, 0], rk[:, None, :, 1] - rk[:, :, None, 1],
          (rk[:, None, :, 0] > rk[:, :, None, 0]).float()]
    return CC, torch.stack(pf, -1)



def extra_features(tr, te):
    """network-level base pair features broadcast to each window, standardised with train statistics only."""
    def feat(d):
        f = base_pair_feats(d)
        return np.stack([np.asarray(f[k], np.float64) for k in sorted(f)], -1).astype(np.float32)
    Ftr = [feat(d) for d in tr]
    Fte = [feat(d) for d in te]
    allf = np.concatenate([f[~np.eye(f.shape[0], dtype=bool)] for f in Ftr])
    mu = allf.mean(0)
    sd = allf.std(0) + 1e-6

    def norm(Fm, W):
        Z = np.clip((Fm - mu) / sd, -5, 5).astype(np.float32)
        Z[np.eye(Fm.shape[0], dtype=bool)] = 0
        return np.repeat(Z[None], W, 0)
    return ([norm(f, d['X'].shape[0]) for f, d in zip(Ftr, tr)],
            [norm(f, d['X'].shape[0]) for f, d in zip(Fte, te)])


def neural_logprobs(tr, te):
    """train the from-scratch neural components on all training networks; mean test head log-probs."""
    P = [prep_net(d) for d in tr]
    PT = [prep_net(d) for d in te]
    ex, ext = extra_features(tr, te)
    comps = []
    for name, lib, cfg in NN_COMPONENTS:
        cfg = json.loads(json.dumps(cfg))
        cfg['eval_at'] = [cfg['epochs']]
        use_ex = lib != 'v1' and bool(cfg.get('extra'))
        if use_ex:
            cfg.setdefault('net', {})['extra'] = ex[0].shape[-1]
        acc = None
        for sd in range(NN_SEEDS):
            if lib == 'v1':
                m, _ = v1_train_model(P, np.arange(len(P)), cfg, seed=1000 + 100 * sd)
                mats = [v1_predict_net(m, p) for p in PT]
            else:
                m, _ = n2_train_model(P, np.arange(len(P)), cfg, ex if use_ex else None, seed=1000 + 100 * sd)
                mats = [n2_predict_net(m, p, ext[i] if use_ex else None) for i, p in enumerate(PT)]
            acc = mats if acc is None else [a + c for a, c in zip(acc, mats)]
        comps.append([n2_norm(a / NN_SEEDS) for a in acc])
        print('neural component', name, 'done', flush=True)
    return [n2_norm(np.mean(x, 0)) for x in zip(*comps)]


# ----------------------------------------------------------------------------- main
def main(public_dir, out_path):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    tr = load_networks(os.path.join(public_dir, 'train.csv'))
    te = load_networks(os.path.join(public_dir, 'test.csv'))
    test_rows = [candidate_rows(d)[0] for d in te]
    gbm = [np.zeros((d['n'], d['n'])) for d in te]
    for b in range(GBM_BAG):
        X, Y, gid = training_set(tr, seed=SEED + b)
        print('bag', b, 'train rows', X.shape, flush=True)
        model = train_model(X, Y, gid, seed=SEED + b)
        for i, d in enumerate(te):
            gbm[i] += head_logprobs(model.predict(test_rows[i], raw_score=True), d['n']) / GBM_BAG
    gbm = [n2_norm(S) for S in gbm]
    nnm = neural_logprobs(tr, te)
    prog = {}
    for d, Sg, Sn in zip(te, gbm, nnm):
        S = n2_norm(Sg + NN_WEIGHT * Sn)
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
