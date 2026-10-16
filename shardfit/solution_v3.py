"""ShardFit: Archaeological Refit Neighborhood Completion - end-to-end solution (v3).

usage: python3 solution_v3.py <public_dataset_dir> <submission_csv_path>

Requirement map
- Card compatibility per candidate: case_feats() builds per-card evidence from the query card and the twelve
  row-local cards (distance/bearing/height/year bands, shared layer, object/production-code agreement with the
  query, masks, within-case ranks and counts) - handles c00..c11 and the s carrier are never used as features.
  fx_extra_feats() adds object-code context (a masked query object code is imputed as the most frequent non-O01
  card object code; agreement with it and code counts) and same-code ranks (d rank among same-p / same-o /
  non-contradicting cards, year rank among same-p cards, number of closer same-p cards).
- Geometry (fx_extra_feats): approximate planar card positions from the distance band and bearing sector
  (radius d+0.5, angle (b+0.5)*45 degrees); pairwise planar and planar+height distances, nearest-card distances
  (all / same p / same o, height gap), counts within radius 1 and 2, mean distance to the 3 nearest cards,
  occlusion counts (cards in the same / adjacent sector closer to the query), distance and relative bearing to
  the card closest to the query, distance to the centroid of the same-p cards with d <= 2.
- Candidate-induced subgraph: case_feats() adds visible degree, typed witness degrees, witness-neighbour
  summaries and witness-component size; fx_extra_feats() adds typed witness-neighbour summaries (neighbour
  distance / d band / same p, same-p R1 and R2 neighbours, 2-hop size, triangles) and hidden-edge evidence:
  excess of the visible degree v over the witness degree and of each h digit over the typed witness degree
  (capped at 2), RX edges assigned to the relation family with the largest positive excess at their ends.
  fx_extra3() adds component competition features (component count, size rank / gap, minimum-d rank, mean d,
  same-p count / fraction / rank, excess sum, d rank inside the component, relative degree) on the witness
  graph (wc_*) and on the hidden-edge-augmented graph (hc_*) that also links every pair of positive-excess
  cards sharing a p code or with p masked. stack_feats() adds typed message passing of stage-1 probabilities
  over the witness graph (neighbour max/sum edge probability per relation family, neighbour type posteriors,
  component maxima) and cross-card competition (row ranks, gaps, sums).
- Pair statistics, computed inside every feat stage-1 fit from that fit's training cases only:
  * nested target encodings (fx_te_features): smoothed (FX_TE_M = 20, towards the fit-set class prior) edge /
    R0 / R1 / R2 rates of the keys (query p, card p), (query p, card p, card d band), (query p, card p, card
    year band); the fit cases get inner 5-fold out-of-fold encodings (fold seed 7), the apply cases (outer
    out-of-fold cases or test) get encodings from all fit cases; unseen keys -> prior.
  * label-free witness p-pair rates (fx_wit_pair_feats): per (query p, card p) code pair, smoothed rate of
    visible witness edges of each type (R0, R1, R2, RX) and their sum among candidate pairs with those codes.
- Learned models trained in this script on train.csv only:
  * stack: stage 1 LightGBM multiclass [none,R0,R1,R2] per candidate on case_feats() (train_stage1, 5-fold OOF
    so stage 2 is trained on out-of-fold inputs); stage 2 LightGBM on the stage-1 probabilities and graph
    context (train_stage2).
  * feat model (train_feat): stage 1 LightGBM multiclass (15 leaves, min_data_in_leaf 50, feature_fraction 0.5,
    lr 0.03, 500 rounds) on the 132 raw features of fx_build() (case_feats + fx_extra_feats + fx_extra3) plus 12
    target encodings and 5 witness p-pair rates; 5-fold (fold seed SEED) out-of-fold probabilities for train and
    a model fit on all train for test. Contextual stage 2 LightGBM (7 leaves, min_data_in_leaf 150,
    feature_fraction 0.5, 300 rounds, averaged over seeds FX_S2SEEDS = 0, 1, 2) on the 132 raw features plus 62
    context features of the stage-1 probabilities (fx_ctx_feats: own probabilities / logit / row rank / gaps,
    case sums / top-3 / counts / type shares, witness-neighbour maximum and sums (per witness type and per
    relation family), 2-hop maximum, witness-component sum, geometric kernel sums, nearest-card probability,
    same-p maximum / sum / rank, hidden-edge-augmented component sum / max / mean / rank / gap, relations to
    the top-1 card, and similarity to likely neighbours: edge-probability-weighted and top-3 means of same p,
    bearing / height / d band / year band differences and planar distance); fit on all train with the out-of-fold stage-1 context and applied
    to the test context from the all-train stage-1 model.
  * neural set model (SetNet, trained from scratch here, nothing pretrained): a permutation-equivariant set
    transformer over 13 tokens - the query token plus the 12 card tokens, with no positional encoding, so it
    is equivariant to the handle order. Card tokens sum embeddings of the card fields (d b z y l o p v h,
    h digits, same_o / same_p with the query, clipped differences between the visible degree / relation
    histogram and the typed witness degrees), a linear map of numeric values + mask flags, an MLP of the
    case_feats() hand features (minus the opaque code ids, standardised with train statistics) and the
    query-field embeddings; the query token sums the query-field embeddings. Witness edges and
    pair relations (witness type, relative bearing, same p / same o, |d| and |z| band gaps, common witness
    neighbours, witness shortest-path length) and query<->card relations (d band, same_o, same_p) enter
    every layer as learned per-head attention biases. Heads: per-card 4-way [none,R0,R1,R2] softmax on
    (card, query) states plus an auxiliary neighbourhood-size (1..6) head on the query token (loss weight
    NN_CFG['aux']). AdamW + warmup/cosine schedule, EMA weights; fit on all train with 3 fixed seeds
    (NN_SEEDS) and the seed test probabilities are averaged (train_neural).
  * Vocabularies, encodings and standardisation statistics come from train only (unknown -> -1 for the GBM
    codes, -> 1 for the neural embeddings, -> fit-set prior for the target encodings and p-pair rates).
- Ensemble: the feat stage-2, stack stage-2 and neural test probabilities are blended by an equal-weight
  geometric mean (exp of the mean of the three log probabilities, each clipped at 1e-9, renormalised over the
  4 classes).
- Neighbourhood-size calibration and constrained decoding: eu_decode() chooses, per case, the number of
  incident edges (1..6) and their relation families by maximising the expected official score under the
  learned probabilities (expected Dice, expected Levenshtein fidelity, linearised pooled per-relation F1);
  class rates and F1 operating points are estimated from training labels / stack stage-2 out-of-fold
  predictions and then applied to the blended test probabilities.
- Output: tokens cNN:Rk sorted by handle, 1-6 tokens, never NONE.
- Determinism: fixed seeds (LightGBM seeds, torch.manual_seed / torch.Generator per neural seed, seeded Monte
  Carlo, fixed fold seeds), fixed thread count (THREADS for BLAS/OpenMP, LightGBM and torch), deterministic
  LightGBM, hash-seed independent code imputation, no clock.
- CPU only.
"""
import json
import os
import re
import sys
from collections import defaultdict

THREADS = 1
os.environ['OMP_NUM_THREADS'] = str(THREADS)
os.environ['MKL_NUM_THREADS'] = str(THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(THREADS)

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as TF

torch.set_num_threads(THREADS)
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
NN_CFG = dict(D=96, H=4, L=3, drop=0.4, ff=2, ep=30, bs=64, lr=2e-3, wd=0.1, warm=2, ema=0.995, aux=0.2)
NN_SEEDS = [0, 1, 2]                               # torch seed 1000 * s + 99 for each



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



# ----------------------------------------------------------------------------- neural set model: tensorisation

NN_CF = ['d', 'b', 'z', 'y', 'l', 'o', 'p', 'v', 'h']    # categorical card fields (never 's', never handles)
NN_WT = {'R0': 1, 'R1': 2, 'R2': 3, 'RX': 4}             # witness pair code (0 = no visible edge, 5 = self)
NN_XCSIZE = [7, 6, 6, 6, 7]                               # clipped degree-difference codes incl. unknown
NN_PSIZE = [6, 9, 3, 3, 8, 14, 3, 6]                      # cardinalities of the 8 pair-relation codes


def nn_build_vocab(cases):
    """embedding vocabularies from TRAIN cases only: index 0 = pad (unused), 1 = unknown, values from 2."""
    V = {('c', k): {} for k in NN_CF}
    V.update({('q', k): {} for k in QFIELDS})
    V.update({('hd', k): {} for k in range(3)})
    for c in cases:
        for card in c['cards']:
            for k in NN_CF:
                V[('c', k)].setdefault(card[k], len(V[('c', k)]) + 2)
            for k in range(3):
                hv = card['h'][k] if card['h'] not in ('M', 'X') else card['h']
                V[('hd', k)].setdefault(hv, len(V[('hd', k)]) + 2)
        for k in QFIELDS:
            V[('q', k)].setdefault(c['q'][k], len(V[('q', k)]) + 2)
    return V


def nn_sizes(V):
    """embedding table sizes: card codes (9 fields, 3 h digits, same_o, same_p, 5 degree differences), query codes."""
    vs = {k: len(v) + 2 for k, v in V.items()}
    csize = np.array([vs[('c', k)] for k in NN_CF] + [vs[('hd', k)] for k in range(3)] + [5, 5] + NN_XCSIZE)
    qsize = np.array([vs[('q', k)] for k in QFIELDS])
    return csize, qsize


def nn_tensorise(cases, V):
    """Ccat (n,12,19) card codes, Cnum (n,12,26) scaled numeric values + NaN flags, Qcat (n,4) query codes,
    Pcat (n,12,12,8) pair-relation codes. Unknown values -> 1."""
    n = len(cases)
    Ccat = np.zeros((n, 12, len(NN_CF) + 3 + 2 + len(NN_XCSIZE)), np.int64)
    Cnum = np.zeros((n, 12, 16 + 10), np.float32)
    Qcat = np.zeros((n, len(QFIELDS)), np.int64)
    Pcat = np.zeros((n, 12, 12, len(NN_PSIZE)), np.int64)
    for i, c in enumerate(cases):
        q, cards = c['q'], c['cards']
        for k, f in enumerate(QFIELDS):
            Qcat[i, k] = V[('q', f)].get(q[f], 1)
        d = np.array([num(x['d']) for x in cards]); y = np.array([num(x['y']) for x in cards])
        v = np.array([num(x['v']) for x in cards]); z = np.array([zval(x['z']) for x in cards])
        h = np.array([[num(x['h'][k]) if x['h'] not in ('M', 'X') else np.nan for k in range(3)] for x in cards])
        b = np.array([num(x['b']) for x in cards])
        for j, card in enumerate(cards):
            for k, f in enumerate(NN_CF):
                Ccat[i, j, k] = V[('c', f)].get(card[f], 1)
            for k in range(3):
                hv = card['h'][k] if card['h'] not in ('M', 'X') else card['h']
                Ccat[i, j, len(NN_CF) + k] = V[('hd', k)].get(hv, 1)
            so = 2 if (card['o'] in ('M', 'X') or q['o'] == 'M') else int(card['o'] == q['o'])
            sp = 2 if (card['p'] in ('M', 'X') or q['p'] == 'M') else int(card['p'] == q['p'])
            Ccat[i, j, len(NN_CF) + 3] = so + 2
            Ccat[i, j, len(NN_CF) + 4] = sp + 2
        sc = [d, y, v, z, np.abs(z), h[:, 0], h[:, 1], h[:, 2]]
        for k, a in enumerate(sc):
            Cnum[i, :, k] = np.nan_to_num(a, nan=0.0) / (6.0 if k in (0, 3, 4) else 3.0)
            Cnum[i, :, 8 + k] = np.isnan(a)
        # pair codes: witness type, bearing difference mod 8, same p, same o, |d| gap, |z| gap
        W = np.zeros((12, 12), np.int64)
        for a_, b_, r in c['wit']:
            W[a_, b_] = NN_WT[r]; W[b_, a_] = NN_WT[r]
        Pcat[i, :, :, 0] = W
        bd = np.where(np.isnan(b[:, None]) | np.isnan(b[None, :]), 8, np.mod(np.nan_to_num(b[:, None]) - np.nan_to_num(b[None, :]), 8)).astype(np.int64)
        Pcat[i, :, :, 1] = bd
        ps = [x['p'] for x in cards]; os_ = [x['o'] for x in cards]
        for a_ in range(12):
            for b_ in range(12):
                Pcat[i, a_, b_, 2] = 2 if (ps[a_] in ('M', 'X') or ps[b_] in ('M', 'X')) else int(ps[a_] == ps[b_])
                Pcat[i, a_, b_, 3] = 2 if (os_[a_] in ('M', 'X') or os_[b_] in ('M', 'X')) else int(os_[a_] == os_[b_])
        dd = np.where(np.isnan(d[:, None]) | np.isnan(d[None, :]), 7, np.abs(np.nan_to_num(d[:, None]) - np.nan_to_num(d[None, :]))).astype(np.int64)
        Pcat[i, :, :, 4] = dd
        zz = np.where(np.isnan(z[:, None]) | np.isnan(z[None, :]), 13, np.clip(np.abs(np.nan_to_num(z[:, None]) - np.nan_to_num(z[None, :])), 0, 12)).astype(np.int64)
        Pcat[i, :, :, 5] = zz
        # 2-hop: number of common witness neighbours (0,1,2+), self-loop marker
        A = (W > 0).astype(int)
        cn = np.clip(A @ A, 0, 2)
        Pcat[i, :, :, 6] = cn
        np.fill_diagonal(Pcat[i, :, :, 0], 5)
        # shortest path distance in witness graph: 0 self,1,2,3,4+ , 5 = disconnected
        Dm = np.full((12, 12), 99); np.fill_diagonal(Dm, 0); Dm[A > 0] = 1
        for kk in range(12):
            Dm = np.minimum(Dm, Dm[:, kk:kk + 1] + Dm[kk:kk + 1, :])
        Pcat[i, :, :, 7] = np.where(Dm >= 99, 5, np.clip(Dm, 0, 4))
        # visible-degree / histogram vs witness typed degree differences (hidden or uncounted edges)
        degr = np.zeros((12, 4))
        for a_, b_, r in c['wit']:
            degr[a_, NN_WT[r] - 1] += 1; degr[b_, NN_WT[r] - 1] += 1
        wdeg = degr.sum(1)
        diffs = [v - wdeg, h[:, 0] - degr[:, 0], h[:, 1] - degr[:, 1], h[:, 2] - degr[:, 2], v - h.sum(1)]
        lo = [-2, -2, -2, -2, -2]; hi = [3, 2, 2, 2, 3]
        for k, a in enumerate(diffs):
            cl = np.clip(np.nan_to_num(a, nan=0), lo[k], hi[k]) - lo[k]
            Ccat[i, :, len(NN_CF) + 5 + k] = np.where(np.isnan(a), NN_XCSIZE[k] - 1, cl)
            Cnum[i, :, 16 + k] = np.nan_to_num(a, nan=0.0) / 2.0
            Cnum[i, :, 21 + k] = np.isnan(a)
    return dict(Ccat=Ccat, Cnum=Cnum, Qcat=Qcat, Pcat=Pcat)


def nn_std_hand(Xtr, Xs):
    """standardise hand features with statistics of Xtr (train rows only), clip to +-5; NaN -> 0 plus NaN
    indicators for the columns that have NaN in train. Statistics are reduced over a C-contiguous copy so the
    float32 summation order does not depend on the caller's memory layout."""
    F = np.ascontiguousarray(Xtr).reshape(-1, Xtr.shape[-1])
    mu = np.nanmean(F, 0); sd = np.nanstd(F, 0) + 1e-6
    nancols = np.isnan(F).any(0)
    out = []
    for X in Xs:
        Z = (X - mu) / sd
        Z = np.clip(Z, -5, 5)
        ind = np.isnan(Z[:, :, nancols]).astype(np.float32)
        out.append(np.concatenate([np.nan_to_num(Z, nan=0.0), ind], -1).astype(np.float32))
    return out



# ----------------------------------------------------------------------------- neural set model: network + training


class SetBlock(nn.Module):
    """pre-LayerNorm transformer block; attention logits get an additive per-head bias (B,H,13,13)."""

    def __init__(self, D, H, drop, ff=2):
        super().__init__()
        self.H = H; self.dh = D // H
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D); self.o = nn.Linear(D, D)
        self.ff = nn.Sequential(nn.Linear(D, ff * D), nn.ReLU(), nn.Linear(ff * D, D))
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x, bias):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        a = torch.baddbmm(bias.reshape(B * self.H, T, T), q.reshape(B * self.H, T, self.dh),
                          k.reshape(B * self.H, T, self.dh).transpose(1, 2), alpha=self.dh ** -0.5)
        a = a.softmax(-1)
        o = (a @ v.reshape(B * self.H, T, self.dh)).view(B, self.H, T, self.dh).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.o(o))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


def nn_offsets(sizes):
    return torch.tensor(np.concatenate([[0], np.cumsum(np.asarray(sizes, int))[:-1]]), dtype=torch.long)


class SetNet(nn.Module):
    """permutation-equivariant set transformer over [query token, 12 card tokens] (no positional encoding).
    Card<->card attention biases come from the pair-relation codes (witness edges etc.), query<->card biases
    from the card's d band and same_o / same_p; both are learned per layer and head."""

    def __init__(self, csize, qsize, nnum, nhand, D, H, L, drop, ff=2):
        super().__init__()
        self.L, self.H = L, H
        self.register_buffer('coff', nn_offsets(csize)); self.register_buffer('qoff', nn_offsets(qsize))
        self.register_buffer('poff', nn_offsets(NN_PSIZE))
        self.cemb = nn.Embedding(int(np.sum(csize)), D)           # card code embeddings (summed over codes)
        self.qemb = nn.Embedding(int(np.sum(qsize)), D)           # query token embeddings
        self.qcemb = nn.Embedding(int(np.sum(qsize)), D)          # query codes added onto every card token
        self.nproj = nn.Linear(nnum, D)                           # numeric values + NaN flags
        self.hmlp = nn.Sequential(nn.Linear(nhand, D), nn.ReLU(), nn.Linear(D, D))   # standardised hand features
        self.qtok = nn.Parameter(torch.zeros(D))
        self.pemb = nn.Embedding(int(np.sum(NN_PSIZE)), L * H)    # card<->card bias per layer and head
        nn.init.normal_(self.pemb.weight, std=0.1)
        self.qcsz = [int(csize[0]), 5, 5]                         # query<->card bias from card d band, same_o, same_p
        self.register_buffer('qcoff', nn_offsets(self.qcsz))
        self.qcb = nn.Embedding(int(np.sum(self.qcsz)), L * 2 * H)
        nn.init.normal_(self.qcb.weight, std=0.1)
        self.blocks = nn.ModuleList([SetBlock(D, H, drop, ff) for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 4))  # per-card [none,R0,R1,R2]
        self.aux = nn.Linear(D, 6)                                # neighbourhood size 1..6 from the query token
        self.edrop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    @staticmethod
    def onehot(idx, n):
        """multi-hot (..., n) over the last index axis; multi-hot @ embedding table = sum of the code embeddings."""
        sh = idx.shape
        z = torch.zeros(int(np.prod(sh[:-1])), n)
        z.scatter_(1, idx.reshape(-1, sh[-1]), 1.0)
        return z.view(*sh[:-1], n)

    def biases(self, Pcat, Ccat):
        B = Pcat.shape[0]; L, H = self.L, self.H
        qi = torch.stack([Ccat[..., 0], Ccat[..., 12], Ccat[..., 13]], -1) + self.qcoff
        qc = self.onehot(qi, self.qcb.weight.shape[0]) @ self.qcb.weight  # B,12,2LH
        qc = qc.view(B, 12, L, 2, H)
        row = torch.cat([torch.zeros(B, 1, L, H), qc[:, :, :, 0]], 1)  # B,13,L,H  (query attends to cards)
        cc = (self.onehot(Pcat + self.poff, self.pemb.weight.shape[0]) @ self.pemb.weight).view(B, 12, 12, L, H)
        rest = torch.cat([qc[:, :, None, :, 1], cc], 2)  # B,12,13,L,H  (cards attend to query and cards)
        full = torch.cat([row[:, None], rest], 1)  # B,13,13,L,H
        return full.permute(3, 0, 4, 1, 2).contiguous()  # L,B,H,13,13

    def forward(self, Ccat, Cnum, Qcat, Pcat, Hd):
        x = self.onehot(Ccat + self.coff, self.cemb.weight.shape[0]) @ self.cemb.weight + self.nproj(Cnum)
        x = x + self.hmlp(Hd)
        qi = Qcat + self.qoff
        qv = self.qemb(qi).sum(-2) + self.qtok
        x = x + self.qcemb(qi).sum(-2)[:, None, :]
        x = torch.cat([qv[:, None, :], x], 1)
        x = self.edrop(x)
        bs = self.biases(Pcat, Ccat)
        for l, blk in enumerate(self.blocks):
            x = blk(x, bs[l])
        x = self.lnf(x)
        q = x[:, 0]; c = x[:, 1:]
        logits = self.head(torch.cat([c, q[:, None, :].expand_as(c)], -1))
        return logits, self.aux(q)


def nn_to_t(T, idx, Hs):
    return (torch.from_numpy(T['Ccat'][idx]), torch.from_numpy(T['Cnum'][idx]), torch.from_numpy(T['Qcat'][idx]),
            torch.from_numpy(T['Pcat'][idx]), torch.from_numpy(Hs[idx]))


def nn_fit_predict(T, Htr, Y, Tt, Hte, csize, qsize, cfg, seed):
    """train SetNet from scratch on all training cases (T, Htr, Y) with a fixed seed; return the EMA model's test
    probabilities (n_test,12,4) [none,R0,R1,R2]. Loss = per-card cross-entropy + aux * neighbourhood-size CE."""
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = SetNet(csize, qsize, T['Cnum'].shape[-1], Htr.shape[-1], D=cfg['D'], H=cfg['H'], L=cfg['L'],
                 drop=cfg['drop'], ff=cfg['ff'])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    n = len(Y); bs = cfg['bs']; spe = (n + bs - 1) // bs; tot = spe * cfg['ep']; wu = spe * cfg['warm']
    # linear warmup over cfg['warm'] epochs, cosine decay to 0 over training
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / max(wu, 1)) * 0.5 * (1 + np.cos(np.pi * min(st, tot) / tot)))
    cw = torch.ones(4)                                     # uniform class weights
    Yt = torch.from_numpy(Y)
    size = torch.clamp((Yt > 0).sum(1) - 1, 0, 5)          # aux target: neighbourhood size 1..6 -> 0..5
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg['ema']))
    net.train()
    for ep in range(cfg['ep']):
        perm = torch.randperm(n, generator=g).numpy()
        for st in range(spe):
            bi = perm[st * bs:(st + 1) * bs]
            lg, la = net(*nn_to_t(T, bi, Htr))
            loss = TF.cross_entropy(lg.reshape(-1, 4), Yt[bi].reshape(-1), weight=cw)
            loss = loss + cfg['aux'] * TF.cross_entropy(la, size[bi])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ema.update_parameters(net)
    net = ema.module
    net.eval()
    nt = len(Tt['Ccat'])
    ps = []
    with torch.no_grad():
        for st in range(0, nt, 512):
            lg, _ = net(*nn_to_t(Tt, np.arange(st, min(st + 512, nt)), Hte))
            ps.append(lg.softmax(-1).numpy())
    return np.concatenate(ps)


def train_neural(tr, te, Y, H, Ht):
    """neural set model: train-only vocabularies and hand-feature statistics; one fit on all training cases per
    seed in NN_SEEDS; returns the seed-averaged test probabilities (n_test,12,4)."""
    V = nn_build_vocab(tr)
    T, Tt = nn_tensorise(tr, V), nn_tensorise(te, V)
    csize, qsize = nn_sizes(V)
    Htr, Hte = nn_std_hand(H, [H, Ht])
    Pn = []
    for s in NN_SEEDS:
        Pn.append(nn_fit_predict(T, Htr, Y, Tt, Hte, csize, qsize, NN_CFG, seed=1000 * s + 99).astype(np.float64))
        print('neural seed %d done' % s, flush=True)
    return np.mean(Pn, 0)



# ----------------------------------------------------------------------------- feat model: evidence features
# per-candidate evidence features = case_feats() + fx_extra_feats() (object-code context, hidden-edge excess,
# RX type inference, same-code ranks, approximate geometry, typed witness-neighbour summaries) + fx_extra3()
# (component competition on the witness graph and on the hidden-edge-augmented graph). Never handles, never s.

FX_RI = {'R0': 0, 'R1': 1, 'R2': 2, 'RX': 3}


def fx_rank_in(vals, mask):
    """rank of vals[i] among vals[mask] (strictly less count); nan if not mask[i]."""
    out = np.full(len(vals), np.nan)
    vv = np.where(np.isnan(vals), 99, vals)
    for i in range(len(vals)):
        if mask[i]:
            out[i] = ((vv < vv[i]) & mask).sum()
    return out


def fx_extra_feats(case, vocab):
    q, cards, wit = case['q'], case['cards'], case['wit']
    n = 12
    F = {}
    d = np.array([num(c['d']) for c in cards])
    b = np.array([num(c['b']) for c in cards])
    z = np.array([zval(c['z']) for c in cards])
    v = np.array([num(c['v']) for c in cards])
    hm = np.array([c['h'] in ('M', 'X') for c in cards])
    h = np.array([[np.nan] * 3 if c['h'] in ('M', 'X') else [float(c['h'][k]) for k in range(3)] for c in cards])
    os_ = [c['o'] for c in cards]; ps = [c['p'] for c in cards]
    # ---- object code context: a masked query object code is imputed as the most frequent non-O01 card code
    # (count ties -> lexicographically smallest code; iteration over sorted codes keeps it hash-seed independent)
    non1 = [o for o in os_ if o not in ('M', 'O01')]
    qo = q['o']
    if qo == 'M':
        qo_imp = max(sorted(set(non1)), key=non1.count) if non1 else 'O01'
    else:
        qo_imp = qo
    F['qo_imp'] = np.full(n, vocab['qo'].get(qo_imp, -1), float)
    F['qo_imp_is1'] = np.full(n, float(qo_imp == 'O01'))
    F['n_non1_case'] = np.full(n, float(len(non1)))
    F['n_o1_case'] = np.full(n, float(sum(o == 'O01' for o in os_)))
    F['same_o_imp'] = np.array([np.nan if o == 'M' else float(o == qo_imp) for o in os_])
    so = np.array([o == qo_imp for o in os_])
    F['n_same_o_case'] = np.full(n, float(so.sum()))
    # ---- witness graph typed
    deg = np.zeros(n); degr = np.zeros((n, 4)); adj = [[] for _ in range(n)]
    for a, bb, r in wit:
        ri = FX_RI[r]; deg[a] += 1; deg[bb] += 1; degr[a, ri] += 1; degr[bb, ri] += 1
        adj[a].append((bb, ri)); adj[bb].append((a, ri))
    for k in range(4):
        F['case_wit_R%d' % k] = np.full(n, degr[:, k].sum() / 2)
    F['case_h0'] = np.full(n, np.nansum(h[:, 0])); F['case_h1'] = np.full(n, np.nansum(h[:, 1])); F['case_h2'] = np.full(n, np.nansum(h[:, 2]))
    # ---- excess (hidden edges): v - witness degree, h_r - typed witness degree (capped at 2)
    F['ex_v'] = v - deg
    F['ex_v_pos'] = (v - deg > 0).astype(float); F['ex_v_pos'][np.isnan(v)] = np.nan
    exh = h - np.minimum(degr[:, :3], 2)
    F['ex_h0'] = exh[:, 0]; F['ex_h1'] = exh[:, 1]; F['ex_h2'] = exh[:, 2]
    F['ex_hsum'] = exh.sum(1) - degr[:, 3]
    F['ex_any'] = ((F['ex_hsum'] > 0) | (F['ex_v'] > 0)).astype(float)
    F['n_ex_case'] = np.full(n, np.nansum(F['ex_any']))
    # inferred RX types: an RX edge is assigned to the relation family with the largest positive excess at its ends
    rxt = np.zeros((n, 3))
    for a, bb, r in wit:
        if r != 'RX':
            continue
        sc_ = np.zeros(3)
        for e in (a, bb):
            if not hm[e]:
                sc_ += np.maximum(exh[e], 0)
        if sc_.sum() > 0:
            t = int(np.argmax(sc_)); rxt[a, t] += 1; rxt[bb, t] += 1
    for k in range(3):
        F['wdeg_inf_R%d' % k] = degr[:, k] + rxt[:, k]
    # ---- same-code ranks
    sp = np.array([p == q['p'] and p != 'M' for p in ps])
    F['drank_samep'] = fx_rank_in(d, sp)
    F['drank_sameo'] = fx_rank_in(d, so)
    F['drank_notdiff'] = fx_rank_in(d, np.array([not ((p != 'M' and q['p'] != 'M' and p != q['p']) or (o != 'M' and o != qo_imp)) for p, o in zip(ps, os_)]))
    F['n_samep_closer'] = np.array([np.sum(sp & (np.nan_to_num(d, nan=99) < (d[i] if np.isfinite(d[i]) else 99))) for i in range(n)], float)
    F['yrank_samep'] = fx_rank_in(np.array([num(c['y']) for c in cards]), sp)
    # ---- geometry: approximate planar position from the distance band (radius d+0.5) and bearing sector
    r = d + 0.5
    th = (b + 0.5) * np.pi / 4
    x = r * np.cos(th); yy = r * np.sin(th)
    pos_ok = np.isfinite(x)
    D = np.sqrt((x[:, None] - x[None]) ** 2 + (yy[:, None] - yy[None]) ** 2)
    np.fill_diagonal(D, np.nan)
    Dz = np.abs(z[:, None] - z[None]); np.fill_diagonal(Dz, np.nan)
    D3 = np.sqrt(D ** 2 + (0.5 * Dz) ** 2)
    db = np.abs(b[:, None] - b[None]) % 8; db = np.minimum(db, 8 - db); np.fill_diagonal(db, np.nan)

    def nanmin_rows(M, mask=None):
        out = np.full(n, np.nan)
        for i in range(n):
            row = M[i] if mask is None else np.where(mask, M[i], np.nan)
            if np.isfinite(row).any():
                out[i] = np.nanmin(row)
        return out
    F['nn_dist'] = nanmin_rows(D)
    F['nn_dist3'] = nanmin_rows(D3)
    F['nn_dist_samep'] = nanmin_rows(D, sp)
    F['nn_dist_sameo'] = nanmin_rows(D, so)
    F['nn_dz'] = nanmin_rows(Dz)
    F['n_within1'] = np.nansum(D < 1.0, 1).astype(float)
    F['n_within2'] = np.nansum(D < 2.0, 1).astype(float)
    F['n_within2_samep'] = np.nansum((D < 2.0) & sp[None], 1).astype(float)
    srt = np.sort(np.where(np.isfinite(D), D, np.inf), 1)
    F['mean3_dist'] = np.where(np.isfinite(srt[:, 2]), srt[:, :3].mean(1), np.nan)
    F['x'] = x; F['yy'] = yy
    # occlusion: candidates in same / adjacent sector closer to query
    dd = np.nan_to_num(d, nan=99)
    F['n_block0'] = np.array([np.sum((db[i] == 0) & (dd < dd[i])) if np.isfinite(b[i]) else np.nan for i in range(n)], float)
    F['n_block1'] = np.array([np.sum((db[i] <= 1) & (dd < dd[i])) if np.isfinite(b[i]) else np.nan for i in range(n)], float)
    F['n_block0_samep'] = np.array([np.sum((db[i] == 0) & (dd < dd[i]) & sp) if np.isfinite(b[i]) else np.nan for i in range(n)], float)
    F['n_same_b_samep'] = np.array([np.sum((db[i] == 0) & sp) if np.isfinite(b[i]) else np.nan for i in range(n)], float)
    # distance to the card closest to the query and its relative bearing
    i0 = int(np.argmin(np.where(np.isfinite(d), d, 99)))
    F['dist_to_closest'] = D[:, i0]; F['db_to_closest'] = db[:, i0]
    # distance to the centroid of same-p cards with d <= 2
    m = sp & pos_ok & (dd <= 2)
    if m.sum() >= 1:
        cx, cy = x[m].mean(), yy[m].mean()
        F['dist_centroid_samep'] = np.sqrt((x - cx) ** 2 + (yy - cy) ** 2)
    else:
        F['dist_centroid_samep'] = np.full(n, np.nan)
    # ---- witness neighbour summaries (typed)
    F['nb_min_dist'] = np.array([np.nanmin([D[i, j] for j, _ in adj[i]]) if adj[i] and np.isfinite([D[i, j] for j, _ in adj[i]]).any() else np.nan for i in range(n)])
    F['nb_mean_d'] = np.array([np.nanmean([d[j] for j, _ in adj[i]]) if adj[i] and np.isfinite([d[j] for j, _ in adj[i]]).any() else np.nan for i in range(n)])
    F['nb_frac_samep'] = np.array([np.mean([sp[j] for j, _ in adj[i]]) if adj[i] else np.nan for i in range(n)])
    F['nb_n_samep_R2'] = np.array([sum(1 for j, t in adj[i] if sp[j] and t == 2) for i in range(n)], float)
    F['nb_n_samep_R1'] = np.array([sum(1 for j, t in adj[i] if sp[j] and t == 1) for i in range(n)], float)
    F['nb_ex_any'] = np.array([sum(1 for j, _ in adj[i] if F['ex_any'][j] > 0) for i in range(n)], float)
    F['nb_d01'] = np.array([sum(1 for j, _ in adj[i] if dd[j] <= 1) for i in range(n)], float)
    # 2-hop size
    F['n_2hop'] = np.array([len({k for j, _ in adj[i] for k, _ in adj[j]} - {i}) for i in range(n)], float)
    # triangles
    nbs = [set(j for j, _ in adj[i]) for i in range(n)]
    F['n_tri'] = np.array([sum(len(nbs[i] & nbs[j]) for j in nbs[i]) / 2 for i in range(n)], float)
    names = sorted(F)
    X = np.stack([np.asarray(F[k], float) for k in names], 1)
    return X, names


def fx_components(A):
    n = len(A); comp = -np.ones(n, int); cid = 0
    for s in range(n):
        if comp[s] >= 0: continue
        st = [s]; comp[s] = cid
        while st:
            u = st.pop()
            for w in np.where(A[u] > 0)[0]:
                if comp[w] < 0: comp[w] = cid; st.append(w)
        cid += 1
    return comp


def fx_comp_feats(A, d, sp, ex, pref):
    """component competition: size / min-distance / same-p-count ranks of a card's component among components
    with >= 2 cards, component mean distance, excess sum, rank of the card's distance inside its component."""
    n = 12; F = {}
    comp = fx_components(A); deg = A.sum(1)
    sizes = np.bincount(comp)
    csz = sizes[comp].astype(float)
    dd = np.where(np.isnan(d), 7, d)
    cids = np.unique(comp)
    cmin = {c: dd[comp == c].min() for c in cids}; cmean = {c: dd[comp == c].mean() for c in cids}
    csp = {c: sp[comp == c].sum() for c in cids}; cex = {c: ex[comp == c].sum() for c in cids}
    big = [c for c in cids if sizes[c] >= 2]
    F[pref + 'n_comps2'] = np.full(n, float(len(big)))
    F[pref + 'max_comp'] = np.full(n, float(sizes.max()))
    F[pref + 'comp_size_rank'] = np.array([sum(1 for c in big if sizes[c] > sizes[comp[i]]) for i in range(n)], float)
    F[pref + 'comp_size_gap'] = csz - sizes.max()
    F[pref + 'comp_mind_rank'] = np.array([sum(1 for c in big if cmin[c] < cmin[comp[i]]) if sizes[comp[i]] >= 2 else np.nan for i in range(n)], float)
    F[pref + 'comp_meand'] = np.array([cmean[comp[i]] for i in range(n)])
    F[pref + 'comp_nsp'] = np.array([csp[comp[i]] for i in range(n)], float)
    F[pref + 'comp_fsp'] = F[pref + 'comp_nsp'] / csz
    F[pref + 'comp_nsp_rank'] = np.array([sum(1 for c in big if csp[c] > csp[comp[i]]) if sizes[comp[i]] >= 2 else np.nan for i in range(n)], float)
    F[pref + 'comp_ex'] = np.array([cex[comp[i]] for i in range(n)], float)
    F[pref + 'size'] = csz
    F[pref + 'd_rank_in_comp'] = np.array([np.sum((comp == comp[i]) & (dd < dd[i])) for i in range(n)], float)
    F[pref + 'deg_rel'] = deg / np.maximum(csz - 1, 1)
    return F


def fx_extra3(case):
    """component features on the raw witness graph (wc_*) and on the hidden-edge-augmented graph (hc_*), which
    additionally links every pair of cards that both have positive excess (visible degree v or histogram total
    above the witness degree) and share their p code (or have p masked)."""
    q, cards, wit = case['q'], case['cards'], case['wit']
    n = 12
    d = np.array([num(c['d']) for c in cards])
    v = np.array([num(c['v']) for c in cards])
    h = np.array([[np.nan] * 3 if c['h'] in ('M', 'X') else [float(c['h'][k]) for k in range(3)] for c in cards])
    sp = np.array([c['p'] == q['p'] and c['p'] != 'M' for c in cards])
    A = np.zeros((n, n)); degr = np.zeros((n, 4))
    for a, b, r in wit:
        A[a, b] = A[b, a] = 1; ri = ['R0', 'R1', 'R2', 'RX'].index(r); degr[a, ri] += 1; degr[b, ri] += 1
    deg = A.sum(1)
    exv = np.nan_to_num(v - deg, nan=0); exh = np.nan_to_num(h.sum(1) - degr.sum(1), nan=0)
    ex = np.maximum(np.maximum(exv, exh), 0)
    F = fx_comp_feats(A, d, sp, ex, 'wc_')
    A2 = A.copy(); exi = np.where(ex > 0)[0]
    ps = [c['p'] for c in cards]
    for i in exi:
        for j in exi:
            if i < j and (ps[i] == ps[j] or 'M' in (ps[i], ps[j])):
                A2[i, j] = A2[j, i] = 1
    F.update(fx_comp_feats(A2, d, sp, ex, 'hc_'))
    F['n_ex_pairs'] = np.full(n, float(len(exi)))
    names = sorted(F)
    return np.stack([np.asarray(F[k], float) for k in names], 1), names


def fx_build(cases, vocab):
    """(n_cases,12,132) feat-model raw features: case_feats + fx_extra_feats + fx_extra3 (train-only vocabulary)."""
    Xs = []
    for c in cases:
        X1, n1 = case_feats(c, vocab)
        X2, n2 = fx_extra_feats(c, vocab)
        X3, n3 = fx_extra3(c)
        Xs.append(np.concatenate([X1, X2, X3], 1))
    return np.stack(Xs), n1 + n2 + n3



# ----------------------------------------------------------------------------- feat model: nested target encodings
# and label-free witness p-pair rates (both computed only from the cases a stage-1 model is fit on)

FX_KN = ['qp_cp', 'qo_co', 'qp_cp_d', 'qe_cy', 'qp_d', 'cp_d', 'qp_cp_z', 'qp_cp_y']
FX_TE_KEYS = ('qp_cp', 'qp_cp_d', 'qp_cp_y')      # (query p, card p), (+ card d band), (+ card year band)
FX_TE_M = 20.0                                      # smoothing towards the fit-set class prior
FX_TE_INNER, FX_TE_SEED = 5, 7                      # inner folds giving out-of-fold encodings to training rows
FX_WP_NAMES = ['wp_R0', 'wp_R1', 'wp_R2', 'wp_RX', 'wp_all']


def fx_keys_for(cases):
    """(n,12,8) object array of code-pair keys (columns FX_KN); raw strings, looked up in fit-set dictionaries."""
    out = []
    for c in cases:
        q = c['q']; rows = []
        for cd in c['cards']:
            zs = 'M' if cd['z'] in ('M', 'X') else ('0' if cd['z'] == '0' else cd['z'][0])
            rows.append((q['p'] + '|' + cd['p'], q['o'] + '|' + cd['o'], q['p'] + '|' + cd['p'] + '|' + cd['d'], q['e'] + '|' + cd['y'],
                         q['p'] + '|' + cd['d'], cd['p'] + '|' + cd['d'], q['p'] + '|' + cd['p'] + '|' + zs, q['p'] + '|' + cd['p'] + '|' + cd['y']))
        out.append(rows)
    return np.array(out, dtype=object)


def fx_te_stats(K, Y):
    """K (n,12,nk) keys; Y (n,12) -> list of dicts key -> [count, class counts (4)]."""
    st = []
    T = np.eye(4)[Y.reshape(-1)]
    Kf = K.reshape(-1, K.shape[2])
    for j in range(K.shape[2]):
        dct = defaultdict(lambda: np.zeros(5))
        for kk, t in zip(Kf[:, j], T):
            a = dct[kk]; a[0] += 1; a[1:] += t
        st.append(dict(dct))
    return st


def fx_te_apply(st, K, prior, m):
    """smoothed class rates (counts + m * prior) / (n + m); unseen key -> prior. Per key: edge, R0, R1, R2 rates."""
    n = K.shape[0]; Kf = K.reshape(-1, K.shape[2]); out = []
    for j, dct in enumerate(st):
        vals = np.zeros((len(Kf), 4))
        for i, kk in enumerate(Kf[:, j]):
            a = dct.get(kk)
            if a is None:
                vals[i] = prior
            else:
                vals[i] = (a[1:] + m * prior) / (a[0] + m)
        out.append(1 - vals[:, 0]); out += [vals[:, 1], vals[:, 2], vals[:, 3]]
    return np.stack(out, 1).reshape(n, 12, -1)


def fx_te_names(KN):
    return ['te_%s_%s' % (k, t) for k in KN for t in ('e', 'r0', 'r1', 'r2')]


def fx_te_features(Ka, Ya, Kb, m):
    """nested target encoding: training rows (Ka, Ya) get inner FX_TE_INNER-fold out-of-fold encodings; apply
    rows Kb get encodings from all of (Ka, Ya)."""
    prior = np.bincount(Ya.reshape(-1), minlength=4) / Ya.size
    fo = folds(len(Ka), FX_TE_INNER, FX_TE_SEED)
    Ta = None
    for j in range(FX_TE_INNER):
        st = fx_te_stats(Ka[fo != j], Ya[fo != j])
        tj = fx_te_apply(st, Ka[fo == j], prior, m)
        if Ta is None:
            Ta = np.zeros((len(Ka), 12, tj.shape[2]))
        Ta[fo == j] = tj
    st = fx_te_stats(Ka, Ya)
    Tb = fx_te_apply(st, Kb, prior, m)
    return Ta, Tb


def fx_wit_pair_stats(cases):
    """label-free: for unordered candidate p-code pairs, candidate-pair counts and visible witness edge counts per
    type (R0, R1, R2, RX) over the given (training) cases."""
    cnt = defaultdict(float); et = defaultdict(lambda: np.zeros(4))
    for c in cases:
        ps = [cd['p'] for cd in c['cards']]
        for i in range(12):
            for j in range(i + 1, 12):
                if 'M' in (ps[i], ps[j]): continue
                cnt[tuple(sorted((ps[i], ps[j])))] += 1
        for a, b, r in c['wit']:
            if 'M' in (ps[a], ps[b]): continue
            et[tuple(sorted((ps[a], ps[b])))][['R0', 'R1', 'R2', 'RX'].index(r)] += 1
    return cnt, dict(et)


def fx_wit_pair_feats(stats, cases, m=20.0):
    """per card: smoothed witness edge rates per type (and their sum) of the (query p, card p) code pair."""
    cnt, et = stats
    tot = sum(cnt.values()); prior = sum(et.values(), np.zeros(4)) / max(tot, 1)
    out = np.full((len(cases), 12, 5), np.nan)
    for i, c in enumerate(cases):
        qp = c['q']['p']
        for j, cd in enumerate(c['cards']):
            if 'M' in (qp, cd['p']): continue
            k = tuple(sorted((qp, cd['p'])))
            n = cnt.get(k, 0.0); e = et.get(k, np.zeros(4))
            r = (e + m * prior) / (n + m)
            out[i, j, :4] = r; out[i, j, 4] = r.sum()
    return out



# ----------------------------------------------------------------------------- feat model: stage-2 context


def fx_case_struct(case):
    """per-case geometry / graph structures used by the stage-2 context features."""
    q, cards, wit = case['q'], case['cards'], case['wit']
    d = np.array([num(c['d']) for c in cards]); b = np.array([num(c['b']) for c in cards])
    r = d + 0.5; th = (b + 0.5) * np.pi / 4
    x = r * np.cos(th); y = r * np.sin(th)
    D = np.sqrt((x[:, None] - x[None]) ** 2 + (y[:, None] - y[None]) ** 2); np.fill_diagonal(D, np.nan)
    A = np.zeros((12, 12)); T = np.full((12, 12), -1)
    for a, bb, rr in wit:
        A[a, bb] = A[bb, a] = 1; T[a, bb] = T[bb, a] = ['R0', 'R1', 'R2', 'RX'].index(rr)
    sp = np.array([c['p'] == q['p'] and c['p'] != 'M' for c in cards])
    v = np.array([num(c['v']) for c in cards])
    hh = np.array([np.nan if c['h'] in ('M', 'X') else float(sum(int(ch) for ch in c['h'])) for c in cards])
    deg = A.sum(1); ex = np.maximum(np.maximum(np.nan_to_num(v - deg, nan=0), np.nan_to_num(hh - deg, nan=0)), 0)
    Ah = A.copy(); exi = np.where(ex > 0)[0]; ps = [c['p'] for c in cards]
    for i in exi:
        for j in exi:
            if i < j and (ps[i] == ps[j] or 'M' in (ps[i], ps[j])):
                Ah[i, j] = Ah[j, i] = 1
    comph = fx_components(Ah)
    comp = fx_components(A)
    zz = np.array([zval(c['z']) for c in cards]); yv = np.array([num(c['y']) for c in cards])
    return dict(D=D, A=A, T=T, sp=sp, comp=comp, Ah=Ah, comph=comph, b=b.copy(), d=d.copy(), p=ps, z=zz, y=yv)


def fx_ctx_feats(P, S):
    """P (12,4) stage-1 probs of one case; S fx_case_struct -> (12, 62) context features, names (sorted)."""
    F = {}
    pe = 1 - P[:, 0]; pr = P[:, 1:]
    # own probabilities, logit, row rank and gaps; case-level sums / top-3 / counts; type shares
    F['s1_pe'] = pe
    for r in range(3): F['s1_pR%d' % r] = pr[:, r]
    F['s1_lpe'] = np.log(pe + 1e-6) - np.log(1 - pe + 1e-6)
    o = np.argsort(-pe); rk = np.empty(12); rk[o] = np.arange(12)
    F['s1_rank'] = rk
    srt = pe[o]
    F['s1_gap_top'] = pe - srt[0]
    F['s1_gap_next'] = np.array([pe[i] - (srt[int(rk[i]) + 1] if rk[i] < 11 else 0) for i in range(12)])
    F['s1_gap_prev'] = np.array([(srt[int(rk[i]) - 1] if rk[i] > 0 else 1) - pe[i] for i in range(12)])
    F['s1_sum'] = np.full(12, pe.sum()); F['s1_max'] = np.full(12, srt[0]); F['s1_2nd'] = np.full(12, srt[1]); F['s1_3rd'] = np.full(12, srt[2])
    F['s1_n05'] = np.full(12, float((pe > 0.5).sum())); F['s1_n03'] = np.full(12, float((pe > 0.3).sum()))
    F['s1_share'] = pe / pe.sum()
    tm = pr.sum(0) / max(pe.sum(), 1e-9)
    for r in range(3): F['s1_case_t%d' % r] = np.full(12, tm[r])
    ownt = pr / np.maximum(pe[:, None], 1e-9)
    for r in range(3): F['s1_own_t%d' % r] = ownt[:, r]
    # witness neighbours (typed), 2-hop, witness component, geometric kernels, nearest card, same-p cards
    A, T, D, sp, comp = S['A'], S['T'], S['D'], S['sp'], S['comp']
    deg = A.sum(1)
    F['nb_pe_max'] = np.where(deg > 0, (A * pe[None]).max(1), np.nan)
    F['nb_pe_sum'] = (A * pe[None]).sum(1)
    for t in range(4):
        F['nb_pe_sum_T%d' % t] = ((T == t) * pe[None]).sum(1)
    for r in range(3):
        F['nb_pr_sum_R%d' % r] = (A * pr[:, r][None]).sum(1)
    A2 = ((A @ A) > 0).astype(float); np.fill_diagonal(A2, 0); A2 = A2 * (1 - A)
    F['nb2_pe_max'] = np.where(A2.sum(1) > 0, (A2 * pe[None]).max(1), np.nan)
    same_c = (comp[:, None] == comp[None]).astype(float); np.fill_diagonal(same_c, 0)
    F['comp_pe_sum'] = (same_c * pe[None]).sum(1)
    Dn = np.nan_to_num(D, nan=99)
    F['geo_pe1'] = (np.exp(-Dn) * pe[None]).sum(1)
    F['geo_pe2'] = (np.exp(-Dn / 2) * pe[None]).sum(1)
    j0 = o[0]; j1 = o[1]
    F['dist_top1'] = np.where(np.arange(12) == j0, D[:, j1], D[:, j0])
    nn_ = np.array([np.nanargmin(D[i]) if np.isfinite(D[i]).any() else -1 for i in range(12)])
    F['nn_pe'] = np.array([pe[nn_[i]] if nn_[i] >= 0 else np.nan for i in range(12)])
    spo = sp[None] * (1 - np.eye(12))
    F['sp_pe_max'] = np.where(spo.sum(1) > 0, (spo * pe[None]).max(1), np.nan)
    F['sp_pe_sum'] = (spo * pe[None]).sum(1)
    F['sp_rank'] = np.array([np.sum(sp & (pe > pe[i])) for i in range(12)], float)
    # hidden-edge-augmented graph components; relations to the top-1 card
    Ah, comph = S['Ah'], S['comph']
    sch = (comph[:, None] == comph[None]).astype(float); np.fill_diagonal(sch, 0)
    F['hc_pe_sum'] = (sch * pe[None]).sum(1)
    F['hc_pe_max'] = np.where(sch.sum(1) > 0, (sch * pe[None]).max(1), np.nan)
    F['hc_pe_mean'] = np.where(sch.sum(1) > 0, (sch * pe[None]).sum(1) / np.maximum(sch.sum(1), 1), np.nan)
    F['hnb_pe_max'] = np.where(Ah.sum(1) > 0, (Ah * pe[None]).max(1), np.nan)
    cs = {c: pe[comph == c].sum() for c in np.unique(comph)}
    csz = np.bincount(comph)
    big = [c for c in cs if csz[c] >= 2]
    F['hc_pesum_rank'] = np.array([sum(1 for c in big if cs[c] > cs[comph[i]]) if csz[comph[i]] >= 2 else np.nan for i in range(12)], float)
    F['hc_pesum_gap'] = np.array([cs[comph[i]] - max([cs[c] for c in big if c != comph[i]], default=0) for i in range(12)], float)
    top = j0
    F['top_same_comp'] = np.where(np.arange(12) == top, np.nan, (comph == comph[top]).astype(float))
    F['top_adj'] = np.where(np.arange(12) == top, np.nan, Ah[:, top])
    F['top_same_p'] = np.where(np.arange(12) == top, np.nan, np.array([float(S['p'][i] == S['p'][top]) if 'M' not in (S['p'][i], S['p'][top]) else np.nan for i in range(12)]))
    bb = S['b']; dbt = np.abs(bb - bb[top]) % 8; dbt = np.minimum(dbt, 8 - dbt)
    F['top_db'] = np.where(np.arange(12) == top, np.nan, dbt)
    F['top_dd'] = np.where(np.arange(12) == top, np.nan, S['d'] - S['d'][top])
    F['top_pe'] = np.full(12, pe[top])
    # pe-weighted mean distance to the other likely neighbours
    w = pe[None] * (1 - np.eye(12)); Dn2 = np.where(np.isfinite(D), D, np.nan)
    F['wdist_pe'] = np.nansum(w * Dn2, 1) / np.maximum(np.nansum(w * np.isfinite(Dn2), 1), 1e-9)
    # similarity to likely neighbours: pe-weighted means (ln_*) and top-3-card means (t3_*) of same p, bearing /
    # height / distance-band / year-band differences and planar distance
    w = pe[None] * (1 - np.eye(12))
    pp = S['p']
    Mp = np.array([[float(pp[i] == pp[j]) if 'M' not in (pp[i], pp[j]) else np.nan for j in range(12)] for i in range(12)])

    def wmean(M):
        ok = np.isfinite(M) & (w > 0)
        return np.where(ok.any(1), np.nansum(np.where(ok, w * M, 0), 1) / np.maximum(np.where(ok, w, 0).sum(1), 1e-9), np.nan)
    F['ln_samep'] = wmean(Mp)
    dbm = np.abs(bb[:, None] - bb[None]) % 8; dbm = np.minimum(dbm, 8 - dbm)
    F['ln_db'] = wmean(dbm)
    F['ln_dz'] = wmean(np.abs(S['z'][:, None] - S['z'][None]))
    F['ln_dd'] = wmean(np.abs(S['d'][:, None] - S['d'][None]))
    F['ln_dy'] = wmean(np.abs(S['y'][:, None] - S['y'][None]))
    F['ln_dist'] = wmean(D)
    o3 = o[:3]; w3 = np.zeros((12, 12)); w3[:, o3] = 1; np.fill_diagonal(w3, 0)

    def hmean(M):
        ok = np.isfinite(M) & (w3 > 0)
        return np.where(ok.any(1), np.nansum(np.where(ok, M, 0), 1) / np.maximum(ok.sum(1), 1), np.nan)
    F['t3_samep'] = hmean(Mp); F['t3_dist'] = hmean(D); F['t3_dz'] = hmean(np.abs(S['z'][:, None] - S['z'][None]))
    names = sorted(F)
    return np.stack([F[k] for k in names], 1), names


def fx_build_ctx(P, structs):
    out = []
    for i in range(len(P)):
        Xc, names = fx_ctx_feats(P[i], structs[i]); out.append(Xc)
    return np.stack(out), names



# ----------------------------------------------------------------------------- feat model: two-stage LightGBM

FX_S1PRM = dict(objective='multiclass', num_class=4, learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.5,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=THREADS, seed=SEED,
                deterministic=True, force_col_wise=True)
FX_PRM2 = dict(learning_rate=0.03, num_leaves=7, min_data_in_leaf=150, feature_fraction=0.5)   # stage 2 overrides
FX_R1, FX_R2 = 500, 300
FX_S2SEEDS = (0, 1, 2)


def fx_catidx(names):
    return [names.index(k) for k in names if k.startswith('c_') or k.startswith('q_') or k == 'qo_imp']


def fx_fit_predict(X, Y, Xb, names, prm, rounds, seed):
    """LightGBM multiclass on (X, Y) with FX_S1PRM updated by prm and the given seed; probabilities on Xb."""
    p = dict(FX_S1PRM); p.update(prm or {}); p['seed'] = seed
    ds = lgb.Dataset(X.reshape(-1, X.shape[2]), Y.reshape(-1), categorical_feature=fx_catidx(names), free_raw_data=False)
    m = lgb.train(p, ds, num_boost_round=rounds)
    return m.predict(Xb.reshape(-1, Xb.shape[2])).reshape(-1, 12, 4)


def fx_stage1(X, K, Y, tr, names, tri, Xb, Kb, cases_b):
    """stage-1 model fit on training cases tri: raw features + nested target encodings (fx_te_features) + witness
    p-pair rates (statistics from the tri cases only); returns probabilities for the apply rows (Xb, Kb, cases_b)."""
    use = [FX_KN.index(k) for k in FX_TE_KEYS]
    ta, tb = fx_te_features(K[tri][:, :, use], Y[tri], Kb[:, :, use], m=FX_TE_M)
    stt = fx_wit_pair_stats([tr[i] for i in tri])
    Xa = np.concatenate([X[tri], ta, fx_wit_pair_feats(stt, [tr[i] for i in tri])], 2)
    Xb = np.concatenate([Xb, tb, fx_wit_pair_feats(stt, cases_b)], 2)
    nm = list(names) + fx_te_names([FX_KN[u] for u in use]) + FX_WP_NAMES
    return fx_fit_predict(Xa, Y[tri], Xb, nm, None, FX_R1, seed=0)


def train_feat(tr, te, Y, vocab):
    """feat two-stage LightGBM -> test probabilities (n_test,12,4). Stage 1: 5-fold (fold seed SEED) out-of-fold
    probabilities for train and a model fit on all train for test. Stage 2: contextual LightGBM on the raw
    features + fx_ctx_feats of the stage-1 probabilities, fit on all train with the out-of-fold stage-1 context,
    applied to the test context; averaged over FX_S2SEEDS."""
    X, names = fx_build(tr, vocab)
    Xt, _ = fx_build(te, vocab)
    K, Kt = fx_keys_for(tr), fx_keys_for(te)
    n = len(tr)
    fo = folds(n, 5, SEED)
    oof = np.zeros((n, 12, 4))
    for k in range(5):
        tri = np.where(fo != k)[0]; vai = np.where(fo == k)[0]
        oof[vai] = fx_stage1(X, K, Y, tr, names, tri, X[vai], K[vai], [tr[i] for i in vai])
    P1t = fx_stage1(X, K, Y, tr, names, np.arange(n), Xt, Kt, te)
    print('feat stage 1 done', flush=True)
    C, cn = fx_build_ctx(oof, [fx_case_struct(c) for c in tr])
    Ct, _ = fx_build_ctx(P1t, [fx_case_struct(c) for c in te])
    Xa, Xb, nm = np.concatenate([X, C], 2), np.concatenate([Xt, Ct], 2), names + cn
    P2t = 0
    for sd in FX_S2SEEDS:
        P2t = P2t + fx_fit_predict(Xa, Y, Xb, nm, FX_PRM2, FX_R2, seed=sd) / len(FX_S2SEEDS)
    print('feat stage 2 done', flush=True)
    return P2t


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
    # decoder operating points learned from training labels / stage-2 out-of-fold predictions
    gold_rate = np.array([(Y == r + 1).sum() for r in range(3)], float) / len(Y)
    theta = np.full(3, 0.25)
    for _ in range(2):
        s, f1 = metric(eu_decode(P2, gold_rate, theta), Y)
        theta = f1 / 2
    s, f1 = metric(eu_decode(P2, gold_rate, theta), Y)
    print('stage-2 train OOF score %.4f  F1 %s  theta %s' % (s, np.round(f1, 4), np.round(theta, 4)), flush=True)
    # feat two-stage LightGBM (evidence features, nested target encodings, witness p-pair rates, contextual stage 2)
    Pf = train_feat(tr, te, Y, vocab)
    # neural set model; its hand features are case_feats() without the opaque code ids c_* / q_* (those codes
    # enter the network as embeddings instead)
    keep = [k for k, nm in enumerate(names) if not nm.startswith(('c_', 'q_'))]
    Pn = train_neural(tr, te, Y, X[:, :, keep].astype(np.float32), Xt[:, :, keep].astype(np.float32))
    print('neural done', flush=True)
    # equal-weight geometric blend of the feat stage-2, stack stage-2 and neural test probabilities (each clipped
    # at 1e-9), renormalised over the 4 classes
    P = np.exp((np.log(np.clip(Pf, 1e-9, 1)) + np.log(np.clip(P2t, 1e-9, 1)) + np.log(np.clip(Pn, 1e-9, 1))) / 3)
    P /= P.sum(2, keepdims=True)
    L = eu_decode(P, gold_rate, theta)
    sub = pd.DataFrame({'id': dft['id'], 'insertion_program': [program(row) for row in L]})
    assert sub['id'].is_unique and len(sub) == len(dft)
    sub.to_csv(out_path, index=False)
    print('wrote', out_path, len(sub), 'rows; mean tokens %.3f' % (L > 0).sum(1).mean(), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
