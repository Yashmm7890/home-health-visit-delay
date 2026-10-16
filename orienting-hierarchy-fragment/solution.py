"""Orienting a Hierarchy Fragment - end-to-end solution.

usage: python3 solution.py <public_dir> <submission_csv>

Pipeline (everything is trained in this script on train.csv + the training rows of terms.csv):
  1. Edge models over token codes, each antisymmetric in the pair (score(a,b) = -score(b,a)):
     a. PairLR      - logistic regression on unigram code differences x_a - x_b plus antisymmetric
                      code-pair features (u in bag_a minus bag_b, v in bag_b minus bag_a).
     b. UniLR       - logistic regression on unigram code differences only.
     c. KronSVM     - SVM with an antisymmetric Kronecker kernel over idf-weighted Tanimoto
                      similarities of the two symmetric-difference code sets.
     d. KronEmbSVM  - KronSVM plus an antisymmetric Kronecker kernel over mean 512-d code embeddings
                      of the symmetric differences (generalises to code pairs never seen in training).
     e. EmbLR       - logistic regression on mean 256-d embedding differences and their
                      antisymmetric outer product.
     f. KNNVote(td) - instance-based analog vote: softmax-weighted forward-vs-backward Kronecker
                      similarity to every oriented training relation (idf-weighted Tanimoto).
     g. KNNVote(emb)- the same vote with cosine similarity of idf-weighted mean embeddings.
     Code embeddings are a PPMI-SVD and idf weights are document frequencies, both fitted on
     training term bags only.
  2. Out-of-fold (5-fold, whole fragments held out) scores of these models on the training
     fragments; test fragments are scored by the same models refitted on all training fragments.
  3. LightGBM combiner (trained on the out-of-fold scores, itself out-of-fold for training items):
     per pair it sees the seven scores, the PairLR unigram/pair decomposition and seen-pair
     fraction, per-term context features of both endpoints (from the item's own bags + training
     document frequencies), each model's mean score against the item's other terms, and one-hop
     context (scores of the other pairs touching each endpoint). It is antisymmetrised by scoring
     both orientations: s = (g(fwd) - g(bwd)) / 2.
  4. Fragment CRF: every one of the 2^k orientations of an item's k listed pairs is enumerated,
     directed cycles are excluded, and an orientation is scored by learned weights on
       - the combiner score and the UniLR score of every pair,
       - structural features (degree categories, number of roots/sinks, 2-path motifs),
       - per-term context features x degree-category potentials.
     Weights are fitted by maximum likelihood on the training fragments (L-BFGS); the CRF
     L2 strength is chosen inside the script by fragment-level 5-fold CV on training items.
  5. Each test pair is oriented by its CRF marginal (orientation probability summed over all
     acyclic orientations of the whole fragment), which is the edge-accuracy optimal decision.

Requirement map:
  - Output: one row per test item, `item_id,directions`, `parent>child` joined by `;`, every
    listed pair exactly once                                   -> write_submission()
  - Learn which tokens signal generality from training fragments -> PairLR/UniLR/KronSVM/
    KronEmbSVM/EmbLR/KNNVote + LightGBM combiner
  - Compare the two terms of a pair rather than scoring a term alone -> antisymmetric pair
    features / kernels in all edge models
  - Use the fragment as a whole (consistent generality across pairs) -> one-hop context in the
    combiner + fragment CRF over all joint orientations, acyclicity, shared per-term potentials
  - Term-disjoint validation holding out whole fragments           -> fragment_folds()
  - No external data, no pretrained models, no hash inversion       -> only files in public_dir
  - No pooling of statistics across test items: every test item is featurised from its own
    rows plus statistics learned from training items only (vocabularies, document
    frequencies, embeddings and model weights are all fitted on training items)
  - No hard-coded per-item answers: all outputs come from the trained CRF.
Determinism: fixed seeds and fold assignment, sorted iteration over every set/dict of codes,
fixed BLAS thread count, no clock or environment reads, CPU only.
"""
import os

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import sys
import collections
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import svds
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb

SEED = 0
N_FOLDS = 5
PAIR_C = 1.0          # PairLR inverse L2 strength (same scale for unigram and pair features)
UNI_C = 1.0
SVM_C = 1.0
SVM_W_UNI = 0.02      # weight of the linear unigram kernel inside KronSVM
EMB_DIM = 256
EMB_OUTER = 64        # leading embedding dims used in the antisymmetric outer-product features
EMB_OUTER_W = 3.0
EMB_C = 1.0
KEMB_DIM = 512       # embedding size used inside KronEmbSVM
KEMB_W = 2.0          # weight of the embedding Kronecker kernel in KronEmbSVM
KEMB_WL = 0.1         # weight of the linear embedding-difference kernel in KronEmbSVM
KNN_TAU = 0.04        # softmax temperature of the analog-edge vote
GBM_ROUNDS = 300
GBM_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=4, seed=0, deterministic=True, force_col_wise=True)
CRF_L2T = 1.0         # structural weights
CRF_L2N = 1.0         # node x degree-category weights
CRF_L2B_GRID = [0.1, 1.0]   # edge-feature weights, chosen by in-script CV
DEG_CAP = 3
DF_EDGES = [0, 1, 3, 10, 30, 100, 300, 10 ** 9]


# ----------------------------------------------------------------------------------- data
def load(D):
    terms = pd.read_csv(os.path.join(D, 'terms.csv'), dtype=str)
    tr = pd.read_csv(os.path.join(D, 'train.csv'), dtype=str)
    te = pd.read_csv(os.path.join(D, 'test.csv'), dtype=str)
    bags = collections.defaultdict(dict)
    for r in terms.itertuples():
        bags[r.item_id][r.term_id] = sorted(set(r.tokens.split()))
    items = []
    for r in tr.itertuples():
        gold = {}
        for d in r.directions.split(';'):
            a, b = d.split('>')
            gold[tuple(sorted((a, b)))] = a
        ps = [tuple(sorted(p.split('-'))) for p in r.pairs.split()]
        items.append(dict(id=r.item_id, pairs=ps, bags=bags[r.item_id], gold=gold))
    titems = []
    for r in te.itertuples():
        raw = r.pairs.split()
        ps = [tuple(sorted(p.split('-'))) for p in raw]
        titems.append(dict(id=r.item_id, pairs=ps, bags=bags[r.item_id]))
    return items, titems, te


def fragment_folds(n, k, seed):
    rng = np.random.RandomState(seed)
    f = np.zeros(n, int)
    for i, j in enumerate(rng.permutation(n)):
        f[j] = i % k
    return f


def all_pairs(it):
    ns = sorted(it['bags'])
    return [(ns[i], ns[j]) for i in range(len(ns)) for j in range(i + 1, len(ns))]


def labels(it):
    return np.array([1 if it['gold'][p] == p[0] else 0 for p in it['pairs']])


# ----------------------------------------------------------------------------- edge models
def pair_feature_dict(A, B):
    """antisymmetric unigram + code-pair features of the ordered pair (A, B)."""
    d = {}
    for t in A:
        d[('u', t)] = d.get(('u', t), 0) + 1
    for t in B:
        d[('u', t)] = d.get(('u', t), 0) - 1
    Ad = sorted(set(A) - set(B)); Bd = sorted(set(B) - set(A))
    for u in Ad:
        for v in Bd:
            if u < v:
                d[('p', u, v)] = d.get(('p', u, v), 0) + 1
            else:
                d[('p', v, u)] = d.get(('p', v, u), 0) - 1
    return d


def sparse_from_dicts(dicts, vocab, grow, keyfilter=None):
    r, c, v = [], [], []
    for i, d in enumerate(dicts):
        for key in sorted(d):
            if keyfilter is not None and key[0] not in keyfilter:
                continue
            j = vocab.get(key)
            if j is None:
                if not grow:
                    continue
                j = vocab[key] = len(vocab)
            r.append(i); c.append(j); v.append(float(d[key]))
    return sp.csr_matrix((v, (r, c)), shape=(len(dicts), len(vocab)))


class DictLR:
    """Antisymmetric logistic regression on feature dicts (no intercept, mirrored training)."""

    def __init__(self, C, keyfilter):
        self.C = C; self.keyfilter = keyfilter

    def fit(self, dicts, y):
        self.vocab = {}
        X = sparse_from_dicts(dicts, self.vocab, True, self.keyfilter)
        m = LogisticRegression(C=self.C, fit_intercept=False, max_iter=5000)
        m.fit(sp.vstack([X, -X]).tocsr(), np.r_[y, 1 - y])
        self.w = m.coef_[0]
        self.is_uni = np.zeros(len(self.vocab), bool)
        for key, j in self.vocab.items():
            self.is_uni[j] = key[0] == 'u'
        return self

    def score(self, dicts):
        X = sparse_from_dicts(dicts, self.vocab, False, self.keyfilter)
        s = X @ self.w
        su = X[:, self.is_uni] @ self.w[self.is_uni]
        seen = []
        for d in dicts:
            pk = [k for k in d if k[0] == 'p']
            seen.append(np.mean([k in self.vocab for k in sorted(pk)]) if pk else 0.0)
        return s, su, np.array(seen)


def train_pairs(items):
    pb, y = [], []
    for it in items:
        for p, lab in zip(it['pairs'], labels(it)):
            pb.append((it['bags'][p[0]], it['bags'][p[1]])); y.append(lab)
    return pb, np.array(y)


class PairModel:
    def __init__(self, C, keyfilter):
        self.lr = DictLR(C, keyfilter)

    def fit(self, items):
        pb, y = train_pairs(items)
        self.lr.fit([pair_feature_dict(a, b) for a, b in pb], y)
        return self

    def score(self, pb):
        return self.lr.score([pair_feature_dict(a, b) for a, b in pb])


class KronSVM:
    """SVM with antisymmetric Kronecker kernel on Tanimoto similarity of symmetric differences."""

    def __init__(self, C, w_uni, emb=None, idf=None, w_emb=KEMB_W, w_embl=KEMB_WL):
        self.C = C; self.w_uni = w_uni; self.emb = emb; self.idf = idf
        self.w_emb = w_emb; self.w_embl = w_embl

    def _rep(self, pb, grow):
        rows = {k: ([], []) for k in ['A', 'B', 'Ad', 'Bd']}
        sizes = {k: [] for k in rows}
        wsz = {k: [] for k in rows}
        for i, (A, B) in enumerate(pb):
            sA, sB = set(A), set(B)
            for key, s in [('A', sA), ('B', sB), ('Ad', sA - sB), ('Bd', sB - sA)]:
                sizes[key].append(len(s))
                if self.idf is not None:
                    wsz[key].append(sum(idf_weight(self.idf, t) for t in sorted(s)))
                for t in sorted(s):
                    j = self.vocab.get(t)
                    if j is None:
                        if not grow:
                            continue
                        j = self.vocab[t] = len(self.vocab)
                    rows[key][0].append(i); rows[key][1].append(j)
        sizes = {k: np.array(v, float) for k, v in sizes.items()}
        if self.idf is not None:
            sizes['w'] = {k: np.array(v, float) for k, v in wsz.items()}
        if self.emb is not None:
            ev, E = self.emb
            ea, eb, fa, fb = [], [], [], []
            for A, B in pb:
                sA, sB = set(A), set(B)
                ea.append(mean_emb(sA - sB, ev, E)); eb.append(mean_emb(sB - sA, ev, E))
                fa.append(mean_emb(sA, ev, E)); fb.append(mean_emb(sB, ev, E))
            sizes['emb'] = (np.array(ea), np.array(eb), np.array(fa), np.array(fb))
        return rows, sizes

    def _mats(self, rows, n):
        V = len(self.vocab)
        return {k: sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, V)) for k, (r, c) in rows.items()}

    def _kernel(self, M1, n1, M2, n2):
        I = lambda a, b: (M1[a] @ M2[b].T).toarray()

        def tani(a, b):
            if self.idf is not None:  # idf-weighted Tanimoto
                x = ((M1[a] @ self.Wd) @ M2[b].T).toarray()
                return x / np.maximum(n1['w'][a][:, None] + n2['w'][b][None, :] - x, 1e-9)
            x = I(a, b)
            return x / np.maximum(n1[a][:, None] + n2[b][None, :] - x, 1)

        K = self.w_uni * (I('A', 'A') - I('A', 'B') - I('B', 'A') + I('B', 'B'))
        K += tani('Ad', 'Ad') * tani('Bd', 'Bd') - tani('Ad', 'Bd') * tani('Bd', 'Ad')
        if self.emb is not None:
            a1, b1, f1, g1 = n1['emb']; a2, b2, f2, g2 = n2['emb']
            K += self.w_emb * ((a1 @ a2.T) * (b1 @ b2.T) - (a1 @ b2.T) * (b1 @ a2.T))
            K += self.w_embl * ((a1 - b1) @ (a2 - b2).T + (f1 - g1) @ (f2 - g2).T)
        return K

    def fit(self, items):
        pb, y = train_pairs(items)
        self.vocab = {}
        rows, n1 = self._rep(pb, True)
        M1 = self._mats(rows, len(pb))
        if self.idf is not None:
            wv = np.zeros(len(self.vocab))
            for t, j in self.vocab.items():
                wv[j] = idf_weight(self.idf, t)
            self.Wd = sp.diags(wv)
        m = SVC(C=self.C, kernel='precomputed')
        m.fit(self._kernel(M1, n1, M1, n1), 2 * y - 1)
        sv = m.support_
        self.coef = m.dual_coef_[0]
        self.Msv = {k: v[sv] for k, v in M1.items()}
        self.nsv = {}
        for k, v in n1.items():
            if k == 'emb':
                self.nsv[k] = tuple(x[sv] for x in v)
            elif k == 'w':
                self.nsv[k] = {kk: vv[sv] for kk, vv in v.items()}
            else:
                self.nsv[k] = v[sv]
        return self

    def score(self, pb):
        rows, n2 = self._rep(pb, False)
        M2 = self._mats(rows, len(pb))
        return self._kernel(M2, n2, self.Msv, self.nsv) @ self.coef


def fit_idf(items):
    """inverse document frequency over training term bags: log((N+1)/(df+1)); unseen codes get df=0."""
    df = collections.Counter()
    N = 0
    for it in items:
        for v in sorted(it['bags']):
            df.update(set(it['bags'][v])); N += 1
    return (dict(df), N)


def idf_weight(idf, t):
    df, N = idf
    return float(np.log((N + 1) / (df.get(t, 0) + 1)))


class KNNVote:
    """Instance-based analog vote: compares a pair (A,B) with every oriented training relation (P>C)
    by the Kronecker similarity sim(A,P)sim(B,C) (forward) vs sim(A,C)sim(B,P) (backward) and returns a
    softmax-weighted vote. mode 'td': idf-weighted Tanimoto of symmetric differences; mode 'emb': cosine
    of idf-weighted mean embeddings of symmetric differences. Training relations only."""

    def __init__(self, mode, idf, emb=None, tau=KNN_TAU):
        self.mode = mode; self.idf = idf; self.emb = emb; self.tau = tau

    def _rep(self, pb, grow):
        n = len(pb)
        out = {}
        for key in ['Ad', 'Bd']:
            r, c, v, sz = [], [], [], []
            E = []
            for i, (A, B) in enumerate(pb):
                sA, sB = set(A), set(B)
                S = sorted(sA - sB) if key == 'Ad' else sorted(sB - sA)
                ws = [idf_weight(self.idf, t) for t in S]
                sz.append(sum(ws))
                if self.mode == 'emb':
                    ev, Em = self.emb
                    e = np.zeros(Em.shape[1])
                    for t, w in zip(S, ws):
                        if t in ev:
                            e += w * Em[ev[t]]
                    E.append(e / (np.linalg.norm(e) + 1e-9))
                for t, w in zip(S, ws):
                    j = self.vocab.get(t)
                    if j is None:
                        if not grow:
                            continue
                        j = self.vocab[t] = len(self.vocab)
                    r.append(i); c.append(j); v.append(w)
            out[key] = (r, c, v, np.array(sz))
            if self.mode == 'emb':
                out['e' + key] = np.array(E)
        return out

    def _mats(self, rep, n, binary):
        V = len(self.vocab)
        R = {}
        for key in ['Ad', 'Bd']:
            r, c, v, sz = rep[key]
            vals = np.ones(len(r)) if binary else np.array(v, float)
            R[key] = sp.csr_matrix((vals, (r, c)), shape=(n, V))
            R['n' + key] = sz
            if 'e' + key in rep:
                R['e' + key] = rep['e' + key]
        return R

    def fit(self, items):
        pb = []
        for it in items:
            for p in it['pairs']:
                par = it['gold'][p]; ch = p[1] if par == p[0] else p[0]
                pb.append((it['bags'][par], it['bags'][ch]))
        self.vocab = {}
        rep = self._rep(pb, True)
        self.Rt = self._mats(rep, len(pb), True)
        return self

    def _sims(self, Rv):
        Rt = self.Rt
        if self.mode == 'emb':
            c = lambda x, y: np.maximum(Rv['e' + x] @ Rt['e' + y].T, 0)
            return c('Ad', 'Ad') * c('Bd', 'Bd'), c('Ad', 'Bd') * c('Bd', 'Ad')

        def tani(x, y):
            I = (Rv[x] @ Rt[y].T).toarray()
            return I / np.maximum(Rv['n' + x][:, None] + Rt['n' + y][None, :] - I, 1e-9)

        return tani('Ad', 'Ad') * tani('Bd', 'Bd'), tani('Ad', 'Bd') * tani('Bd', 'Ad')

    def score(self, pb):
        out = np.zeros(len(pb))
        for st in range(0, len(pb), 500):
            chunk = pb[st:st + 500]
            Rv = self._mats(self._rep(chunk, False), len(chunk), False)
            f, b = self._sims(Rv)
            M = np.c_[f, b]
            w = np.exp((M - M.max(1, keepdims=True)) / self.tau)
            out[st:st + len(chunk)] = 5 * (w[:, :f.shape[1]].sum(1) - w[:, f.shape[1]:].sum(1)) / w.sum(1)
        return out


def fit_embedding(items, dim):
    """PPMI-SVD code embeddings from co-occurrence inside training term bags (training items only)."""
    vocab = {}
    r, c = [], []
    row = 0
    for it in items:
        for v in sorted(it['bags']):
            for t in it['bags'][v]:
                j = vocab.setdefault(t, len(vocab)); r.append(row); c.append(j)
            row += 1
    B = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(row, len(vocab)))
    Cm = (B.T @ B).tocsr().astype(float)
    Cm.setdiag(0); Cm.eliminate_zeros()
    tot = Cm.sum(); rs = np.asarray(Cm.sum(1)).ravel()
    Cm = Cm.tocoo()
    pmi = np.maximum(np.log(Cm.data * tot / (rs[Cm.row] * rs[Cm.col])), 0)
    M = sp.csr_matrix((pmi, (Cm.row, Cm.col)), shape=Cm.shape)
    U, S, _ = svds(M, k=dim, random_state=SEED)
    order = np.argsort(-S)
    E = U[:, order] * np.sqrt(S[order])
    return vocab, E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def mean_emb(s, vocab, E):
    ix = [vocab[t] for t in sorted(s) if t in vocab]
    return E[ix].mean(0) if ix else np.zeros(E.shape[1])


class EmbLR:
    """LR on differences of mean PPMI-SVD code embeddings plus their antisymmetric outer product."""

    def __init__(self, emb, C):
        self.vocab, self.E = emb; self.C = C; self.dim = self.E.shape[1]

    def _feat(self, A, B):
        sA, sB = set(A), set(B)
        a = mean_emb(sA - sB, self.vocab, self.E); b = mean_emb(sB - sA, self.vocab, self.E)
        O = np.outer(a[:EMB_OUTER], b[:EMB_OUTER]); O = O - O.T
        iu = np.triu_indices(EMB_OUTER, 1)
        return np.r_[a - b, mean_emb(sA, self.vocab, self.E) - mean_emb(sB, self.vocab, self.E), EMB_OUTER_W * O[iu]]

    def fit(self, items):
        pb, y = train_pairs(items)
        X = np.array([self._feat(a, b) for a, b in pb])
        m = LogisticRegression(C=self.C, fit_intercept=False, max_iter=5000)
        m.fit(np.r_[X, -X], np.r_[y, 1 - y])
        self.w = m.coef_[0]
        return self

    def score(self, pb):
        return np.array([self._feat(a, b) for a, b in pb]) @ self.w


def fit_edge_models(tr_items):
    emb = fit_embedding(tr_items, EMB_DIM)
    emb_k = fit_embedding(tr_items, KEMB_DIM)
    idf = fit_idf(tr_items)
    return dict(pair=PairModel(PAIR_C, None).fit(tr_items),
                uni=PairModel(UNI_C, {'u'}).fit(tr_items),
                svm=KronSVM(SVM_C, SVM_W_UNI, idf=idf).fit(tr_items),
                kemb=KronSVM(SVM_C, SVM_W_UNI, emb_k, idf=idf).fit(tr_items),
                emb=EmbLR(emb, EMB_C).fit(tr_items),
                knn=KNNVote('td', idf).fit(tr_items),
                knne=KNNVote('emb', idf, emb).fit(tr_items))


def score_items(models, its):
    """per item: dict with listed-pair scores (k x 4), decomposition, all-pair scores (m x 4)."""
    out = []
    for it in its:
        res = {}
        for name, pairs in [('edge', it['pairs']), ('all', all_pairs(it))]:
            pb = [(it['bags'][a], it['bags'][b]) for a, b in pairs]
            s_pair, s_uni_part, seen = models['pair'].score(pb)
            s_uni = models['uni'].score(pb)[0]
            cols = [s_pair, s_uni, models['svm'].score(pb), models['kemb'].score(pb), models['emb'].score(pb),
                    models['knn'].score(pb), models['knne'].score(pb)]
            res[name] = np.stack(cols, 1)
            if name == 'edge':
                res['decomp'] = (s_uni_part, s_pair - s_uni_part, seen)
        out.append(res)
    return out


# --------------------------------------------------------------------- per-item term features
def item_df(items):
    df = collections.Counter()
    for it in items:
        s = set()
        for bag in it['bags'].values():
            s.update(bag)
        df.update(s)
    return df


def node_feats(it, df, own):
    """context features of each term, from this item's own bags + training document frequency.
    own=True for training items (their own occurrence is removed from df)."""
    nodes = sorted(it['bags'])
    sets = {v: set(it['bags'][v]) for v in nodes}
    nb = collections.defaultdict(set)
    for a, b in it['pairs']:
        nb[a].add(b); nb[b].add(a)
    allc = collections.Counter()
    for v in nodes:
        allc.update(sets[v])
    H = {}
    for v in nodes:
        S = sets[v]; L = len(S); St = sorted(S); nbl = sorted(nb[v])
        spread0 = sum(1 for t in St if allc[t] == 1) / L
        innb = sum(1 for t in St if any(t in sets[u] for u in nbl)) / L
        allnb = sum(1 for t in St if all(t in sets[u] for u in nbl)) / L
        jac = np.mean([len(S & sets[u]) / len(S | sets[u]) for u in nbl])
        coh = [len(sets[x] & sets[y]) / len(sets[x] | sets[y]) for i, x in enumerate(nbl) for y in nbl[i + 1:]]
        coh = np.mean(coh) if coh else 0.0
        nonnb = [u for u in nodes if u != v and u not in nb[v]]
        innon = sum(1 for t in St if any(t in sets[u] for u in nonnb)) / L if nonnb else 0.0
        dfs = np.array([df.get(t, 0) - (1 if own else 0) for t in St])
        bucket = np.histogram(dfs, bins=DF_EDGES)[0] / L
        ldf = np.mean(np.log1p(dfs))
        H[v] = np.r_[L / 22.0, spread0, innb, allnb, jac, coh, innon, bucket, ldf, len(nbl) / 4.0]
    return H


def ctx_feats(it, la):
    """from all-pair scores: per-edge mean score difference against the item's other terms,
    and per-term mean score against all other terms."""
    ns = sorted(it['bags']); m = len(ns); idx = {v: i for i, v in enumerate(ns)}
    Lm = np.zeros((m, m)); q = 0
    for i in range(m):
        for j in range(i + 1, m):
            Lm[i, j] = la[q]; Lm[j, i] = -la[q]; q += 1
    G = Lm.sum(1) / (m - 1)
    Z = []
    for a, b in it['pairs']:
        ia, ib = idx[a], idx[b]
        others = [c for c in range(m) if c not in (ia, ib)]
        Z.append(np.mean(Lm[ia, others] - Lm[ib, others]) if others else 0.0)
    return np.array(Z)[:, None], G


def gbm_feats(it, sc, H):
    """per-edge features for the LightGBM combiner, in both orientations (fwd: first listed as parent).
    Antisymmetric columns are negated and endpoint columns swapped for the reverse orientation."""
    k = len(it['pairs'])
    Ecols = sc['edge']
    uu, pp, ff = sc['decomp']
    anti = [Ecols, np.c_[uu, pp]]
    ga, gb = [], []
    ns = sorted(it['bags']); nidx = {v: q for q, v in enumerate(ns)}
    for c in range(sc['all'].shape[1]):
        zc, G = ctx_feats(it, sc['all'][:, c])
        anti.append(zc)
        ga.append(np.array([[G[nidx[a]]] for a, b in it['pairs']]))
        gb.append(np.array([[G[nidx[b]]] for a, b in it['pairs']]))
    sym = np.c_[ff, np.tile([k / 10.0, len(ns) / 8.0], (k, 1))]
    nbr = collections.defaultdict(list)
    for e, (a, b) in enumerate(it['pairs']):
        nbr[a].append((e, 1.0)); nbr[b].append((e, -1.0))

    def hop(v, excl):
        lst = [(e, sg) for e, sg in nbr[v] if e != excl]
        if not lst:
            return np.r_[np.zeros(2 * Ecols.shape[1]), 0.0]
        M = np.array([Ecols[e] * sg for e, sg in lst])
        return np.r_[M.mean(0), M.max(0), len(lst)]

    ha = np.array([np.r_[H[a], hop(a, e)] for e, (a, b) in enumerate(it['pairs'])])
    hb = np.array([np.r_[H[b], hop(b, e)] for e, (a, b) in enumerate(it['pairs'])])
    A = np.concatenate(anti, 1)
    PA = np.concatenate([ha] + ga, 1); PB = np.concatenate([hb] + gb, 1)
    return np.c_[A, sym, PA, PB], np.c_[-A, sym, PB, PA]


def gbm_fit(feats_list, items_list):
    X, y = [], []
    for (fw, bw), it in zip(feats_list, items_list):
        g = labels(it).astype(float)
        X += [fw, bw]; y += [g, 1 - g]
    return lgb.train(GBM_PARAMS, lgb.Dataset(np.concatenate(X), np.concatenate(y)), num_boost_round=GBM_ROUNDS)


def gbm_score(m, fw, bw):
    return (m.predict(fw, raw_score=True) - m.predict(bw, raw_score=True)) / 2


def crf_inputs(it, gbm_s, sc, H):
    ns = sorted(it['bags'])
    Z = np.c_[gbm_s, sc['edge'][:, 1]]  # combiner score + unigram LR score
    Hm = np.c_[np.array([H[v] for v in ns]), np.ones((len(ns), 1))]
    return Z, Hm


# ------------------------------------------------------------------------------ fragment CRF
NC = 9
NF = (DEG_CAP + 1) ** 2 + 5 + 5 + 3


def enumerate_item(it):
    nodes = sorted(it['bags']); idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes); k = len(it['pairs'])
    codes = np.arange(2 ** k)
    M = ((codes[:, None] >> np.arange(k)[None, :]) & 1).astype(np.int8)  # 1: first listed is parent
    a = np.array([idx[p[0]] for p in it['pairs']]); b = np.array([idx[p[1]] for p in it['pairs']])
    Ia = np.zeros((k, n)); Ia[np.arange(k), a] = 1
    Ib = np.zeros((k, n)); Ib[np.arange(k), b] = 1
    P = M.astype(float); Q = 1 - P
    outd = P @ Ia + Q @ Ib; ind = P @ Ib + Q @ Ia
    A = np.zeros((len(codes), n, n), np.int32)
    for e in range(k):
        A[:, a[e], b[e]] |= (M[:, e] == 1)
        A[:, b[e], a[e]] |= (M[:, e] == 0)
    R = A.copy()
    for _ in range(n):
        R = ((R + np.einsum('oij,ojk->oik', R, A)) > 0).astype(np.int32)
    acyc = ~np.any(np.diagonal(R, axis1=1, axis2=2) > 0, axis=1)
    O = len(codes)
    ci = np.minimum(ind, DEG_CAP).astype(int); co = np.minimum(outd, DEG_CAP).astype(int)
    F_deg = np.zeros((O, (DEG_CAP + 1) ** 2))
    for v in range(n):
        F_deg[np.arange(O), ci[:, v] * (DEG_CAP + 1) + co[:, v]] += 1
    roots = (ind == 0).sum(1); sinks = (outd == 0).sum(1)
    F_r = np.zeros((O, 5)); F_r[np.arange(O), np.minimum(roots, 5) - 1] = 1
    F_s = np.zeros((O, 5)); F_s[np.arange(O), np.minimum(sinks, 5) - 1] = 1
    F_m = np.stack([(ind * outd).sum(1), (ind * (ind - 1) / 2).sum(1), (outd * (outd - 1) / 2).sum(1)], 1) / k
    F = np.concatenate([F_deg, F_r, F_s, F_m], 1)
    cat = (np.minimum(ind, 2) * 3 + np.minimum(outd, 2)).astype(int)
    OH = np.zeros(cat.shape + (NC,)); np.put_along_axis(OH, cat[..., None], 1, axis=2)
    gold = None
    if 'gold' in it:
        gc = int((labels(it) << np.arange(k)).sum())
        gold = int(np.flatnonzero(codes[acyc] == gc)[0])
    return dict(S=2.0 * M[acyc] - 1, F=F[acyc], C=OH[acyc].reshape(int(acyc.sum()), n * NC), gold=gold)


class Batch:
    def __init__(self, E, Z, H):
        self.d = Z[0].shape[1]; self.dh = H[0].shape[1]
        self.S = sp.block_diag([sp.csr_matrix(e['S']) for e in E], format='csr')
        self.F = np.concatenate([e['F'] for e in E])
        self.C = sp.block_diag([sp.csr_matrix(e['C']) for e in E], format='csr')
        self.Z = np.concatenate(Z); self.H = np.concatenate(H)
        sizes = np.array([e['S'].shape[0] for e in E])
        self.starts = np.r_[0, np.cumsum(sizes)[:-1]]
        self.seg = np.repeat(np.arange(len(E)), sizes)
        self.gold = np.array([e['gold'] if e['gold'] is not None else 0 for e in E]) + self.starts
        ks = np.array([e['S'].shape[1] for e in E]); self.koff = np.r_[0, np.cumsum(ks)]
        self.N = len(E)

    def split(self, x):
        return x[:self.d], x[self.d:self.d + NF], x[self.d + NF:].reshape(NC, self.dh)

    def probs(self, x):
        beta, th, TH = self.split(x)
        sc = self.S @ (self.Z @ beta) + self.F @ th + self.C @ (self.H @ TH.T).ravel()
        mx = np.maximum.reduceat(sc, self.starts)
        ex = np.exp(sc - mx[self.seg])
        Zs = np.add.reduceat(ex, self.starts)
        return sc, ex / Zs[self.seg], mx + np.log(Zs)


def crf_nll(x, B, l2b, l2t, l2n):
    sc, p, lz = B.probs(x)
    tot = (lz - sc[B.gold]).sum()
    r = p.copy(); np.subtract.at(r, B.gold, 1.0)
    gb = B.Z.T @ (B.S.T @ r); gt = B.F.T @ r
    gT = (B.C.T @ r).reshape(-1, NC).T @ B.H
    beta, th, TH = B.split(x)
    tot += 0.5 * (l2b * (beta ** 2).sum() + l2t * (th ** 2).sum() + l2n * (TH ** 2).sum())
    return tot, np.r_[gb + l2b * beta, gt + l2t * th, (gT + l2n * TH).ravel()]


def crf_fit(B, l2b):
    x0 = np.zeros(B.d + NF + NC * B.dh)
    r = minimize(crf_nll, x0, args=(B, l2b, CRF_L2T, CRF_L2N), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=3000))
    return r.x


def crf_marginals(x, B):
    _, p, _ = B.probs(x)
    m = (B.S.T @ p + 1) / 2
    return [m[B.koff[i]:B.koff[i + 1]] for i in range(B.N)]


def edge_accuracy(items, margs):
    c = n = 0
    for it, m in zip(items, margs):
        g = labels(it)
        c += int(((m >= 0.5).astype(int) == g).sum()); n += len(g)
    return c / n


# ------------------------------------------------------------------------------------- main
def write_submission(te, titems, margs, path):
    rows = []
    by_id = {it['id']: (it, m) for it, m in zip(titems, margs)}
    for r in te.itertuples():
        it, m = by_id[r.item_id]
        ent = []
        for (a, b), q in zip(it['pairs'], m):
            ent.append(f'{a}>{b}' if q >= 0.5 else f'{b}>{a}')
        rows.append((r.item_id, ';'.join(ent)))
    sub = pd.DataFrame(rows, columns=['item_id', 'directions'])
    sub.to_csv(path, index=False)
    return sub


def main(D, out_path):
    items, titems, te = load(D)
    n = len(items)
    print(f'train items {n}, test items {len(titems)}', flush=True)

    # 1-2. out-of-fold edge-model scores on training fragments
    f = fragment_folds(n, N_FOLDS, SEED)
    oof = [None] * n
    for fo in range(N_FOLDS):
        tr = [items[i] for i in range(n) if f[i] != fo]
        va = [i for i in range(n) if f[i] == fo]
        models = fit_edge_models(tr)
        for i, s in zip(va, score_items(models, [items[i] for i in va])):
            oof[i] = s
        print(f'edge-model fold {fo} done', flush=True)
    edge_acc = np.mean(np.concatenate([(oof[i]['edge'][:, 0] >= 0) == labels(items[i]) for i in range(n)]))
    print(f'PairLR OOF edge score {2 * edge_acc - 1:.4f}', flush=True)
    models = fit_edge_models(items)
    tsc = score_items(models, titems)
    print('edge models refitted on all training fragments', flush=True)

    # 3. LightGBM combiner over edge-model scores + per-term context (out-of-fold on training items)
    df = item_df(items)
    HN = [node_feats(it, df, True) for it in items]
    HT = [node_feats(it, df, False) for it in titems]
    GF = [gbm_feats(it, oof[i], HN[i]) for i, it in enumerate(items)]
    GFT = [gbm_feats(it, tsc[i], HT[i]) for i, it in enumerate(titems)]
    gbm_oof = [None] * n
    for fo in range(N_FOLDS):
        tr = [i for i in range(n) if f[i] != fo]
        m = gbm_fit([GF[i] for i in tr], [items[i] for i in tr])
        for i in range(n):
            if f[i] == fo:
                gbm_oof[i] = gbm_score(m, *GF[i])
    acc = np.mean(np.concatenate([(gbm_oof[i] >= 0) == labels(items[i]) for i in range(n)]))
    print(f'LightGBM combiner OOF edge score {2 * acc - 1:.4f}', flush=True)
    m = gbm_fit(GF, items)
    gbm_test = [gbm_score(m, *GFT[i]) for i in range(len(titems))]

    # 4. CRF inputs
    E = [enumerate_item(it) for it in items]
    ZH = [crf_inputs(it, gbm_oof[i], oof[i], HN[i]) for i, it in enumerate(items)]
    ET = [enumerate_item(it) for it in titems]
    ZHT = [crf_inputs(it, gbm_test[i], tsc[i], HT[i]) for i, it in enumerate(titems)]

    # in-script CV for the CRF edge-weight L2 (fragment-level folds)
    f2 = fragment_folds(n, N_FOLDS, SEED + 1)
    best = None
    for l2b in CRF_L2B_GRID:
        margs = [None] * n
        for fo in range(N_FOLDS):
            tr = [i for i in range(n) if f2[i] != fo]; va = [i for i in range(n) if f2[i] == fo]
            x = crf_fit(Batch([E[i] for i in tr], [ZH[i][0] for i in tr], [ZH[i][1] for i in tr]), l2b)
            for i, m in zip(va, crf_marginals(x, Batch([E[i] for i in va], [ZH[i][0] for i in va], [ZH[i][1] for i in va]))):
                margs[i] = m
        acc = edge_accuracy(items, margs)
        print(f'CRF l2b={l2b}: CV score {2 * acc - 1:.4f}', flush=True)
        if best is None or acc > best[0]:
            best = (acc, l2b)
    print(f'chosen l2b={best[1]}', flush=True)

    # 5. final CRF on all training fragments -> test marginals
    x = crf_fit(Batch(E, [z for z, h in ZH], [h for z, h in ZH]), best[1])
    tm = crf_marginals(x, Batch(ET, [z for z, h in ZHT], [h for z, h in ZHT]))
    sub = write_submission(te, titems, tm, out_path)
    print(f'wrote {len(sub)} rows to {out_path}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
