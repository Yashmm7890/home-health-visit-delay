"""ShardFit: Archaeological Refit Neighborhood Completion - end-to-end solution (v1).

usage: python3 solution.py <public_dataset_dir> <submission_csv_path>

Requirement map
- Card compatibility per candidate: case_feats() builds per-card evidence from the query card and the twelve
  row-local cards (distance/bearing/height/year bands, shared layer, object/production-code agreement with the
  query, masks, within-case ranks and counts) - handles c00..c11 and the s carrier are never used as features.
- Candidate-induced subgraph: case_feats() adds visible degree, typed witness degrees, witness-neighbour
  summaries and witness-component size; stack_feats() adds typed message passing of stage-1 probabilities over
  the witness graph (neighbour max/sum edge probability per relation family, neighbour type posteriors,
  component maxima) and cross-card competition (row ranks, gaps, sums).
- Learned models trained in this script on train.csv only: stage 1 LightGBM multiclass [none,R0,R1,R2] per
  candidate (train_stage1, 5-fold OOF so stage 2 is trained on out-of-fold inputs); stage 2 LightGBM on the
  stage-1 probabilities and graph context (stage 2). Vocabularies come from train only (unknown -> -1).
- Neighbourhood-size calibration and constrained decoding: eu_decode() chooses, per case, the number of
  incident edges (1..6) and their relation families by maximising the expected official score under the
  learned probabilities (expected Dice, expected Levenshtein fidelity, linearised pooled per-relation F1);
  class rates and F1 operating points are estimated from training labels / out-of-fold predictions.
- Output: tokens cNN:Rk sorted by handle, 1-6 tokens, never NONE.
- Determinism: fixed seeds, fixed thread count (THREADS), deterministic LightGBM, seeded Monte Carlo, no clock.
- CPU only.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

THREADS = 1
SEED = 0
W_A, W_B, W_C = 0.45, 0.35, 0.20
RELS = ['R0', 'R1', 'R2']
QFIELDS = ['o', 'p', 'l', 'e']
CARD_RE = re.compile(r'(c\d\d)\[([^\]]*)\]')
WT = ['R0', 'R1', 'R2', 'RX']
PRM1 = dict(objective='multiclass', num_class=4, learning_rate=0.03, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=THREADS, seed=SEED,
            deterministic=True, force_col_wise=True)
PRM2 = dict(objective="multiclass", num_class=4, learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=THREADS, seed=SEED, deterministic=True)
ROUNDS1, ROUNDS2 = 500, 400
EU_SAMPLES = 128



# ----------------------------------------------------------------------------- parsing


def parse_query(q):
    parts = q.strip('[]').split(',')
    out = {}
    for name, t in zip(QFIELDS, parts):
        out[name] = 'M' if t == 'M' else t[1:]
    return out


def parse_cards(s):
    cards = []
    for m in CARD_RE.finditer(s):
        f = {}
        for t in m.group(2).split(','):
            f[t[0]] = t[1:]
        cards.append(f)
    assert len(cards) == 12
    return cards


def parse_witness(w):
    """list of (i, j, rel) with rel in R0/R1/R2/RX."""
    if w == 'NO_WITNESS':
        return []
    out = []
    for e in w.split(' '):
        a, r, b = e.split('-')
        out.append((int(a[1:]), int(b[1:]), r))
    return out


def parse_program(s):
    """-> dict cand -> rel index (0..2)."""
    if s == 'NONE':
        return {}
    out = {}
    for t in s.split(' '):
        c, r = t.split(':')
        out[int(c[1:])] = RELS.index(r)
    return out


def labels_from_programs(progs):
    Y = np.zeros((len(progs), 12), int)  # 0 none, 1..3 = R0..R2
    for i, s in enumerate(progs):
        for c, r in parse_program(s).items():
            Y[i, c] = r + 1
    return Y


def to_program(sel):
    """sel: dict cand -> rel index; sorted by handle."""
    if not sel:
        return 'NONE'
    return ' '.join('c%02d:%s' % (c, RELS[r]) for c, r in sorted(sel.items()))


def folds(n, k=5, seed=0):
    rng = np.random.RandomState(seed)
    f = np.zeros(n, int)
    f[rng.permutation(n)] = np.arange(n) % k
    return f



# ----------------------------------------------------------------------------- stage-1 features


def zval(z):
    if z in ('M', 'X'):
        return np.nan
    if z == '0':
        return 0.0
    return float(z[1:]) * (1 if z[0] == 'P' else -1)


def num(v):
    try:
        return float(v)
    except ValueError:
        return np.nan


def case_feats(case, vocab):
    q, cards, wit = case['q'], case['cards'], case['wit']
    n = 12
    F = {}
    d = np.array([num(c['d']) for c in cards])
    y = np.array([num(c['y']) for c in cards])
    v = np.array([num(c['v']) for c in cards])
    z = np.array([zval(c['z']) for c in cards])
    h = np.array([[num(c['h'][k]) if c['h'] not in ('M', 'X') else np.nan for k in range(3)] for c in cards])
    F['d'] = d; F['y'] = y; F['v'] = v; F['z'] = z; F['absz'] = np.abs(z)
    F['h0'] = h[:, 0]; F['h1'] = h[:, 1]; F['h2'] = h[:, 2]
    for k in ['d', 'b', 'z', 'y', 'l', 'o', 'p', 'v', 'h']:
        F['m_' + k] = np.array([c[k] == 'M' for c in cards], float)
    F['nmask'] = sum(F['m_' + k] for k in ['d', 'b', 'z', 'y', 'l', 'o', 'p', 'v', 'h'])
    for k in ['b', 'o', 'p', 'l']:
        F['c_' + k] = np.array([vocab[k].get(c[k], -1) for c in cards], float)
    F['c_z'] = np.array([vocab['z'].get(c['z'], -1) for c in cards], float)
    for k in ['o', 'p', 'l', 'e']:
        F['q_' + k] = np.full(n, vocab['q' + k].get(q[k], -1), float)
    # query matches
    F['same_o'] = np.array([np.nan if (c['o'] == 'M' or q['o'] == 'M') else float(c['o'] == q['o']) for c in cards])
    F['same_p'] = np.array([np.nan if (c['p'] == 'M' or q['p'] == 'M') else float(c['p'] == q['p']) for c in cards])
    # within-case ranks / counts
    dd = np.where(np.isnan(d), 99, d)
    F['d_rank'] = np.array([(dd < dd[i]).sum() for i in range(n)], float)
    F['d_rank_eq'] = np.array([(dd == dd[i]).sum() for i in range(n)], float)
    F['d_min_case'] = np.full(n, np.nanmin(d) if np.isfinite(d).any() else np.nan)
    F['d_minus_min'] = d - F['d_min_case']
    yy = np.where(np.isnan(y), 99, y)
    F['y_rank'] = np.array([(yy < yy[i]).sum() for i in range(n)], float)
    ps = [c['p'] for c in cards]
    F['n_same_p_in_case'] = np.array([sum(1 for x in ps if x == p and p != 'M') for p in ps], float)
    F['n_qp_in_case'] = np.full(n, sum(1 for x in ps if x == q['p'] and x != 'M'))
    F['n_v_pos_case'] = np.full(n, float(np.nansum(v > 0)))
    # bearing agreement with others (same sector count)
    bs = [c['b'] for c in cards]
    F['n_same_b'] = np.array([sum(1 for x in bs if x == b and b not in ('M', 'X')) for b in bs], float)
    # witness graph
    deg = np.zeros(n); degr = np.zeros((n, 4))
    adj = [[] for _ in range(n)]
    for a, b, r in wit:
        ri = ['R0', 'R1', 'R2', 'RX'].index(r)
        deg[a] += 1; deg[b] += 1; degr[a, ri] += 1; degr[b, ri] += 1
        adj[a].append((b, ri)); adj[b].append((a, ri))
    F['wdeg'] = deg
    for k, nm in enumerate(['R0', 'R1', 'R2', 'RX']):
        F['wdeg_' + nm] = degr[:, k]
    F['n_wit'] = np.full(n, len(wit), float)
    # witness-neighbour summaries
    F['nb_min_d'] = np.array([np.nanmin([d[j] for j, _ in adj[i]]) if adj[i] and np.isfinite([d[j] for j, _ in adj[i]]).any() else np.nan for i in range(n)])
    F['nb_samep'] = np.array([sum(1 for j, _ in adj[i] if cards[j]['p'] == q['p'] and q['p'] != 'M') for i in range(n)], float)
    F['nb_drank_min'] = np.array([min([F['d_rank'][j] for j, _ in adj[i]]) if adj[i] else np.nan for i in range(n)])
    # connected component size in witness graph
    comp = -np.ones(n, int); cid = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        st = [s]; comp[s] = cid
        while st:
            u = st.pop()
            for w_, _ in adj[u]:
                if comp[w_] < 0:
                    comp[w_] = cid; st.append(w_)
        cid += 1
    sizes = np.bincount(comp)
    F['wcomp_size'] = sizes[comp].astype(float)
    F['wcomp_min_d'] = np.array([np.nanmin(np.where(comp == comp[i], d, np.nan)) if np.isfinite(np.where(comp == comp[i], d, np.nan)).any() else np.nan for i in range(n)])
    names = sorted(F)
    X = np.stack([np.asarray(F[k], float) for k in names], 1)
    return X, names


def build_vocab(case_lists):
    vocab = {k: {} for k in ['b', 'z', 'o', 'p', 'l', 'qo', 'qp', 'ql', 'qe']}
    for cases in case_lists:
        for c in cases:
            for card in c['cards']:
                for k in ['b', 'z', 'o', 'p', 'l']:
                    vocab[k].setdefault(card[k], len(vocab[k]))
            for k in ['o', 'p', 'l', 'e']:
                vocab['q' + k].setdefault(c['q'][k], len(vocab['q' + k]))
    return vocab


def build(cases, vocab):
    Xs = []
    for c in cases:
        X, names = case_feats(c, vocab)
        Xs.append(X)
    return np.stack(Xs), names  # (n_cases, 12, F)



# ----------------------------------------------------------------------------- stage-2 graph context


def adj_tensor(cases):
    A = np.zeros((len(cases), 4, 12, 12))
    for i, c in enumerate(cases):
        for a, b, r in c['wit']:
            k = WT.index(r)
            A[i, k, a, b] = 1; A[i, k, b, a] = 1
    return A


def comp_matrix(A):
    """same-component indicator (n,12,12) excluding self."""
    Au = (A.sum(1) > 0).astype(float)
    R = Au + np.eye(12)[None]
    for _ in range(4):
        R = ((R @ R) > 0).astype(float)
    return R - np.eye(12)[None]


def stack_feats(P, A, Comp, Xraw=None, hop2=False):
    n = len(P)
    lp = np.log(np.clip(P, 1e-9, 1))
    pe = 1 - P[:, :, 0]
    q = P[:, :, 1:]
    F = [lp[:, :, 1:] - lp[:, :, :1], pe[..., None], q / np.maximum(pe[..., None], 1e-9)]
    rank = np.argsort(np.argsort(-pe, 1, kind='stable'), 1)
    mx = pe.max(1, keepdims=True)
    srt = -np.sort(-pe, 1)
    F += [rank[..., None], (pe / mx)[..., None], (pe - mx)[..., None], np.repeat(pe.sum(1, keepdims=True), 12, 1)[..., None],
          (pe.sum(1, keepdims=True) - pe)[..., None], np.repeat(srt[:, :4], 12, 0).reshape(n, 12, 4) if False else np.repeat(srt[:, None, :4], 12, 1)]
    Au = (A.sum(1) > 0).astype(float)
    deg = Au.sum(2)
    F.append(deg[..., None])
    for M in [Au] + [A[:, k] for k in range(4)]:
        nbmax = np.where(M > 0, pe[:, None, :], -1).max(2)
        nbsum = (M * pe[:, None, :]).sum(2)
        F += [nbmax[..., None], nbsum[..., None]]
    for r in range(3):
        F.append(np.where(Au > 0, q[:, None, :, r], -1).max(2)[..., None])
    cm = np.where(Comp > 0, pe[:, None, :], -1).max(2)
    F += [cm[..., None], (Comp * pe[:, None, :]).sum(2)[..., None], Comp.sum(2)[..., None]]
    if hop2:
        A2 = ((Au @ Au) > 0).astype(float) * (1 - Au) * (1 - np.eye(12)[None])
        F += [np.where(A2 > 0, pe[:, None, :], -1).max(2)[..., None], (A2 * pe[:, None, :]).sum(2)[..., None],
              ((Au * (pe[:, None, :] > 0.5)).sum(2))[..., None],
              np.where(Au > 0, rank[:, None, :], 12).min(2)[..., None]]
        # neighbour type posterior weighted by neighbour edge prob
        tau = q / np.maximum(pe[..., None], 1e-9)
        wsum = (Au * pe[:, None, :]).sum(2, keepdims=True)
        F.append(np.einsum('nij,nj,njr->nir', Au, pe, tau) / np.maximum(wsum, 1e-9))
    if Xraw is not None:
        F.append(Xraw)
    return np.concatenate(F, 2)



# ----------------------------------------------------------------------------- expected-utility decoder


def _atoms_arr(L):
    """L (n,12) -> padded atom array (n,24) and lengths; handle atoms 0..11, rel atoms 100+r."""
    n = len(L)
    A = np.full((n, 24), -1, int)
    ln = np.zeros(n, int)
    for c in range(12):
        m = L[:, c] > 0
        rows = np.where(m)[0]
        A[rows, ln[rows]] = c
        A[rows, ln[rows] + 1] = 100 + L[rows, c]
        ln[rows] += 2
    return A, ln


def lev_rows(Ga, gl, Pa, pl):
    """row-wise Levenshtein between padded int sequences; vectorised over rows (rows sorted by gold length so
    the active set shrinks as the DP advances)."""
    n = len(Ga)
    if n == 0:
        return np.zeros(0, int)
    srt = np.argsort(-gl, kind='stable')
    Ga, gl, Pa, pl = Ga[srt], gl[srt], Pa[srt], pl[srt]
    Lg = int(gl[0])
    Lp = int(pl.max())
    prev = np.tile(np.arange(Lp + 1), (n, 1))
    res = np.where(gl == 0, pl, 0).astype(int)
    cnt = np.array([(gl >= i).sum() for i in range(Lg + 2)])
    for i in range(1, Lg + 1):
        na = cnt[i]
        pv = prev[:na]
        cur = np.empty_like(pv)
        cur[:, 0] = i
        gi = Ga[:na, i - 1]
        Pn = Pa[:na]
        for j in range(1, Lp + 1):
            np.minimum(np.minimum(pv[:, j], cur[:, j - 1]) + 1, pv[:, j - 1] + (gi != Pn[:, j - 1]), out=cur[:, j])
        m = slice(cnt[i + 1], na)          # rows with gl == i
        res[m] = cur[m, :][np.arange(na - cnt[i + 1]), pl[m]]
        prev = cur
    out = np.empty_like(res)
    out[srt] = res
    return out


def pb_dist(p):
    """Poisson-binomial pmf. p (..., m) -> (..., m+1)."""
    sh = p.shape[:-1]
    m = p.shape[-1]
    d = np.zeros(sh + (m + 1,))
    d[..., 0] = 1.0
    for j in range(m):
        pj = p[..., j:j + 1]
        nd = d * (1 - pj)
        nd[..., 1:] += d[..., :-1] * pj
        d = nd
    return d


def exp_dice_topk(pe_sorted, kmax=6, cond_nonempty=True, r=None):
    """pe_sorted (n,12) edge probs sorted desc. Returns (n,kmax) E[Dice] for top-k sets, k=1..kmax.
    r (n,13) optional reweighting of the gold size m: P(config) ∝ prod Bern * r[m]."""
    n, m = pe_sorted.shape
    out = np.zeros((n, kmax))
    if r is None:
        r = np.ones((n, m + 1))
        if cond_nonempty:
            r[:, 0] = 0.0
    for k in range(1, kmax + 1):
        PX = pb_dist(pe_sorted[:, :k])          # (n,k+1)
        PY = pb_dist(pe_sorted[:, k:])          # (n,m-k+1)
        x = np.arange(k + 1)[:, None]; y = np.arange(m - k + 1)[None, :]
        val = 2 * x / (x + y + k)               # (k+1, m-k+1)
        R = r[:, (x + y)]                       # (n,k+1,m-k+1)
        joint = PX[:, :, None] * PY[:, None, :] * R
        out[:, k - 1] = (joint * val[None]).sum((1, 2)) / np.maximum(joint.sum((1, 2)), 1e-300)
    return out


def sample_gold(P, S=32, seed=0, cond_nonempty=True):
    """independent categorical draws per candidate -> (n,S,12) labels; empty draws redrawn (up to 30 tries)."""
    rng = np.random.RandomState(seed)
    n = len(P)
    cum = np.cumsum(P, 2)[:, :, :3]                       # thresholds for classes 1..3
    u = rng.rand(n, S, 12)
    G = (u[..., None] > cum[:, None, :, :]).sum(3)
    if cond_nonempty:
        for it in range(30):
            ii, ss = np.where((G > 0).sum(2) == 0)
            if len(ii) == 0:
                break
            u = rng.rand(len(ii), 12)
            G[ii, ss] = (u[..., None] > cum[ii]).sum(2)
    return G


def exp_C_topk(order, types, G, kmax=6, wts=None):
    """order (n,12) candidate indices sorted by priority, types (n,12) label 1..3 per candidate,
    G (n,S,12) sampled gold, wts (n,S) optional importance weights. Returns (n,kmax) MC estimate of E[C]."""
    n, S, _ = G.shape
    Ga, gl = _atoms_arr(G.reshape(n * S, 12))
    out = np.zeros((n, kmax))
    if wts is None:
        wts = np.full((n, S), 1.0 / S)
    else:
        wts = wts / np.maximum(wts.sum(1, keepdims=True), 1e-300)
    for k in range(1, kmax + 1):
        L = np.zeros((n, 12), int)
        rows = np.arange(n)[:, None]
        L[rows, order[:, :k]] = types[rows, order[:, :k]]
        Pa, pl = _atoms_arr(np.repeat(L, S, 0))
        lev = lev_rows(Ga, gl, Pa, pl)
        C = 1 - lev / np.maximum(np.maximum(gl, pl), 1)
        out[:, k - 1] = (C.reshape(n, S) * wts).sum(1)
    return out


def _labels_from_k(order, types, kb):
    n = len(order)
    L = np.zeros((n, 12), int)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(12)[None, :].repeat(n, 0), 1)
    m = ranks < kb[:, None]
    L[m] = types[m]
    return L


# ----------------------------------------------------------------------------- pipeline
def load_split(path):
    df = pd.read_csv(path)
    cases = [dict(id=r.id, q=parse_query(r.query_card), cards=parse_cards(r.candidate_stream),
                  wit=parse_witness(r.witness_stream)) for r in df.itertuples(index=False)]
    Y = labels_from_programs(df.insertion_program) if 'insertion_program' in df else None
    return df, cases, Y


def train_stage1(X, Y, Xt, cat, fo):
    """5-fold out-of-fold stage-1 probabilities for train and fold-averaged probabilities for test."""
    n, nt, k = len(X), len(Xt), X.shape[2]
    P = np.zeros((n, 12, 4))
    Pt = np.zeros((nt, 12, 4))
    for f in range(fo.max() + 1):
        trm = fo != f
        m = lgb.train(PRM1, lgb.Dataset(X[trm].reshape(-1, k), Y[trm].reshape(-1), categorical_feature=cat),
                      num_boost_round=ROUNDS1)
        P[~trm] = m.predict(X[~trm].reshape(-1, k)).reshape(-1, 12, 4)
        Pt += m.predict(Xt.reshape(-1, k)).reshape(-1, 12, 4) / (fo.max() + 1)
    return P, Pt


def train_stage2(F, Y, Ft, fo):
    """out-of-fold stage-2 probabilities (used only to estimate decoder operating points) and test probabilities
    from a stage-2 model fit on all training cases."""
    k = F.shape[2]
    P2 = np.zeros((len(F), 12, 4))
    for f in range(fo.max() + 1):
        trm = fo != f
        m = lgb.train(PRM2, lgb.Dataset(F[trm].reshape(-1, k), Y[trm].reshape(-1)), num_boost_round=ROUNDS2)
        P2[~trm] = m.predict(F[~trm].reshape(-1, k)).reshape(-1, 12, 4)
    m = lgb.train(PRM2, lgb.Dataset(F.reshape(-1, k), Y.reshape(-1)), num_boost_round=ROUNDS2)
    return P2, m.predict(Ft.reshape(-1, k)).reshape(-1, 12, 4)


def eu_decode(P, gold_rate, theta, S=EU_SAMPLES, seed=SEED, kmax=6):
    """expected-utility decoding: per case, relation family t_c = argmax_r beta_r (q_cr - theta_r) and the
    number k of top-ranked (by edge probability) candidates maximising
    W_A E[Dice] + W_C E[C] + sum over chosen tokens 2 beta_r (q - theta_r)  (linearised pooled F1 term)."""
    q = P[:, :, 1:]
    pe = 1 - P[:, :, 0]
    order = np.argsort(-pe, 1, kind='stable')
    ED = exp_dice_topk(np.take_along_axis(pe, order, 1), kmax)
    G = sample_gold(P, S, seed)
    beta = (W_B / 3) / (2 * np.asarray(gold_rate, float))
    theta = np.asarray(theta, float)
    types = (beta * 2 * (q - theta)).argmax(2) + 1
    EC = exp_C_topk(order, types, G, kmax)
    qt = np.take_along_axis(q, (types - 1)[..., None], 2)[..., 0]
    qo = np.take_along_axis(qt, order, 1)[:, :kmax]
    to = np.take_along_axis(types, order, 1)[:, :kmax]
    u = 2 * beta[to - 1] * (qo - theta[to - 1])
    J = W_A * ED + W_C * EC + np.cumsum(u, 1)
    return _labels_from_k(order, types, J.argmax(1) + 1)


def metric(L, Y):
    """official score (0-100) and per-relation pooled F1 for label arrays L, Y (n,12) with 0 none, 1..3 R0..R2."""
    Ga, gl = _atoms_arr(Y)
    Pa, pl = _atoms_arr(L)
    lev = lev_rows(Ga, gl, Pa, pl)
    C = np.where(pl > 0, 1 - lev / np.maximum(np.maximum(gl, pl), 1), 0.0)
    Ng, Np = Y > 0, L > 0
    A = 2 * (Ng & Np).sum(1) / np.maximum(Ng.sum(1) + Np.sum(1), 1)
    f1 = np.zeros(3)
    for r in range(3):
        tp = ((L == r + 1) & (Y == r + 1)).sum()
        fp = ((L == r + 1) & (Y != r + 1)).sum()
        fn = ((Y == r + 1) & (L != r + 1)).sum()
        f1[r] = 2 * tp / max(2 * tp + fp + fn, 1)
    return 100 * float(np.clip(W_A * A.mean() + W_B * f1.mean() + W_C * C.mean(), 0, 1)), f1


def program(row):
    toks = ['c%02d:%s' % (c, RELS[row[c] - 1]) for c in range(12) if row[c] > 0]
    assert 1 <= len(toks) <= 6
    return ' '.join(toks)


def main(public_dir, out_path):
    np.random.seed(SEED)
    _, tr, Y = load_split(os.path.join(public_dir, 'train.csv'))
    dft, te, _ = load_split(os.path.join(public_dir, 'test.csv'))
    vocab = build_vocab([tr])                                   # train-only vocabulary
    X, names = build(tr, vocab)
    Xt, _ = build(te, vocab)
    cat = [names.index(k) for k in names if k.startswith('c_') or k.startswith('q_')]
    fo = folds(len(tr), 5, SEED)
    P1, P1t = train_stage1(X, Y, Xt, cat, fo)
    print('stage 1 done', flush=True)
    A, At = adj_tensor(tr), adj_tensor(te)
    F = stack_feats(P1, A, comp_matrix(A))
    Ft = stack_feats(P1t, At, comp_matrix(At))
    P2, P2t = train_stage2(F, Y, Ft, fo)
    print('stage 2 done', flush=True)
    # decoder operating points learned from training labels / out-of-fold predictions
    gold_rate = np.array([(Y == r + 1).sum() for r in range(3)], float) / len(Y)
    theta = np.full(3, 0.25)
    for _ in range(2):
        s, f1 = metric(eu_decode(P2, gold_rate, theta), Y)
        theta = f1 / 2
    s, f1 = metric(eu_decode(P2, gold_rate, theta), Y)
    print('train OOF score %.4f  F1 %s  theta %s' % (s, np.round(f1, 4), np.round(theta, 4)), flush=True)
    L = eu_decode(P2t, gold_rate, theta)
    sub = pd.DataFrame({'id': dft['id'], 'insertion_program': [program(row) for row in L]})
    assert sub['id'].is_unique and len(sub) == len(dft)
    sub.to_csv(out_path, index=False)
    print('wrote', out_path, len(sub), 'rows; mean tokens %.3f' % (L > 0).sum(1).mean(), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
