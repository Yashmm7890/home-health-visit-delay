"""Orienting a Hierarchy Fragment - end-to-end solution.

usage: python3 solution.py <public_dir> <submission_csv>

Pipeline (everything is trained in this script on train.csv + the training rows of terms.csv):
  1. Edge models over token codes, each antisymmetric in the pair (score(a,b) = -score(b,a)):
     a. PairLR  - logistic regression on unigram code differences x_a - x_b plus antisymmetric
                  code-pair features (u in bag_a minus bag_b, v in bag_b minus bag_a).
     b. UniLR   - logistic regression on unigram code differences only.
     c. KronSVM - SVM with an antisymmetric Kronecker kernel over Tanimoto similarities of the
                  two symmetric-difference code sets.
     d. EmbLR   - logistic regression on differences of mean code embeddings; embeddings are a
                  PPMI-SVD fitted on training term bags only.
  2. Out-of-fold (5-fold, whole fragments held out) scores of these models on the training
     fragments become features of a fragment-level CRF; test fragments are scored by the same
     models refitted on all training fragments.
  3. Fragment CRF: every one of the 2^k orientations of an item's k listed pairs is enumerated,
     directed cycles are excluded, and an orientation is scored by learned weights on
       - the edge-model scores (+ seen-fraction interactions of the PairLR decomposition),
       - per-term context features computed from that item's own term bags only,
       - structural features (degree categories, number of roots/sinks, 2-path motifs),
       - per-term features x degree-category potentials.
     Weights are fitted by maximum likelihood on the training fragments (L-BFGS); the CRF
     L2 strength is chosen inside the script by fragment-level 5-fold CV on training items.
  4. Each test pair is oriented by its CRF marginal (orientation probability summed over all
     acyclic orientations of the whole fragment), which is the edge-accuracy optimal decision.

Requirement map:
  - Output: one row per test item, `item_id,directions`, `parent>child` joined by `;`, every
    listed pair exactly once                                   -> write_submission()
  - Learn which tokens signal generality from training fragments -> PairLR/UniLR/KronSVM/EmbLR
  - Compare the two terms of a pair rather than scoring a term alone -> antisymmetric pair
    features / kernels in all edge models
  - Use the fragment as a whole (consistent generality across pairs) -> fragment CRF over all
    joint orientations, acyclicity, shared per-term potentials
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

SEED = 0
N_FOLDS = 5
PAIR_C = 1.0          # PairLR inverse L2 strength (same scale for unigram and pair features)
UNI_C = 1.0
SVM_C = 1.0
SVM_W_UNI = 0.02      # weight of the linear unigram kernel inside KronSVM
EMB_DIM = 64
EMB_C = 1.0
CRF_L2T = 1.0         # structural weights
CRF_L2N = 1.0         # node x degree-category weights
CRF_L2B_GRID = [0.1, 1.0, 3.0]   # edge-feature weights, chosen by in-script CV
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

    def __init__(self, C, w_uni):
        self.C = C; self.w_uni = w_uni

    def _rep(self, pb, grow):
        rows = {k: ([], []) for k in ['A', 'B', 'Ad', 'Bd']}
        sizes = {k: [] for k in rows}
        for i, (A, B) in enumerate(pb):
            sA, sB = set(A), set(B)
            for key, s in [('A', sA), ('B', sB), ('Ad', sA - sB), ('Bd', sB - sA)]:
                sizes[key].append(len(s))
                for t in sorted(s):
                    j = self.vocab.get(t)
                    if j is None:
                        if not grow:
                            continue
                        j = self.vocab[t] = len(self.vocab)
                    rows[key][0].append(i); rows[key][1].append(j)
        return rows, {k: np.array(v, float) for k, v in sizes.items()}

    def _mats(self, rows, n):
        V = len(self.vocab)
        return {k: sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, V)) for k, (r, c) in rows.items()}

    def _kernel(self, M1, n1, M2, n2):
        I = lambda a, b: (M1[a] @ M2[b].T).toarray()

        def tani(a, b):
            x = I(a, b)
            return x / np.maximum(n1[a][:, None] + n2[b][None, :] - x, 1)

        K = self.w_uni * (I('A', 'A') - I('A', 'B') - I('B', 'A') + I('B', 'B'))
        K += tani('Ad', 'Ad') * tani('Bd', 'Bd') - tani('Ad', 'Bd') * tani('Bd', 'Ad')
        return K

    def fit(self, items):
        pb, y = train_pairs(items)
        self.vocab = {}
        rows, n1 = self._rep(pb, True)
        M1 = self._mats(rows, len(pb))
        m = SVC(C=self.C, kernel='precomputed')
        m.fit(self._kernel(M1, n1, M1, n1), 2 * y - 1)
        sv = m.support_
        self.coef = m.dual_coef_[0]
        self.Msv = {k: v[sv] for k, v in M1.items()}
        self.nsv = {k: v[sv] for k, v in n1.items()}
        return self

    def score(self, pb):
        rows, n2 = self._rep(pb, False)
        M2 = self._mats(rows, len(pb))
        return self._kernel(M2, n2, self.Msv, self.nsv) @ self.coef


class EmbLR:
    """LR on differences of mean PPMI-SVD code embeddings (embeddings fitted on training bags)."""

    def __init__(self, dim, C):
        self.dim = dim; self.C = C

    def _embed(self, items):
        self.vocab = {}
        r, c = [], []
        row = 0
        for it in items:
            for v in sorted(it['bags']):
                for t in it['bags'][v]:
                    j = self.vocab.setdefault(t, len(self.vocab)); r.append(row); c.append(j)
                row += 1
        B = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(row, len(self.vocab)))
        Cm = (B.T @ B).tocsr().astype(float)
        Cm.setdiag(0); Cm.eliminate_zeros()
        tot = Cm.sum(); rs = np.asarray(Cm.sum(1)).ravel()
        Cm = Cm.tocoo()
        pmi = np.maximum(np.log(Cm.data * tot / (rs[Cm.row] * rs[Cm.col])), 0)
        M = sp.csr_matrix((pmi, (Cm.row, Cm.col)), shape=Cm.shape)
        U, S, _ = svds(M, k=self.dim, random_state=SEED)
        order = np.argsort(-S)
        E = U[:, order] * np.sqrt(S[order])
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def _feat(self, A, B):
        def mean(s):
            ix = [self.vocab[t] for t in sorted(s) if t in self.vocab]
            return self.E[ix].mean(0) if ix else np.zeros(self.dim)
        sA, sB = set(A), set(B)
        return np.r_[mean(sA - sB) - mean(sB - sA), mean(sA) - mean(sB)]

    def fit(self, items):
        self._embed(items)
        pb, y = train_pairs(items)
        X = np.array([self._feat(a, b) for a, b in pb])
        m = LogisticRegression(C=self.C, fit_intercept=False, max_iter=5000)
        m.fit(np.r_[X, -X], np.r_[y, 1 - y])
        self.w = m.coef_[0]
        return self

    def score(self, pb):
        return np.array([self._feat(a, b) for a, b in pb]) @ self.w


def fit_edge_models(tr_items):
    return dict(pair=PairModel(PAIR_C, None).fit(tr_items),
                uni=PairModel(UNI_C, {'u'}).fit(tr_items),
                svm=KronSVM(SVM_C, SVM_W_UNI).fit(tr_items),
                emb=EmbLR(EMB_DIM, EMB_C).fit(tr_items))


def score_items(models, its):
    """per item: dict with listed-pair scores (k x 4), decomposition, all-pair scores (m x 4)."""
    out = []
    for it in its:
        res = {}
        for name, pairs in [('edge', it['pairs']), ('all', all_pairs(it))]:
            pb = [(it['bags'][a], it['bags'][b]) for a, b in pairs]
            s_pair, s_uni_part, seen = models['pair'].score(pb)
            s_uni = models['uni'].score(pb)[0]
            cols = [s_pair, s_uni, models['svm'].score(pb), models['emb'].score(pb)]
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


def crf_inputs(it, sc, H):
    ns = sorted(it['bags'])
    uu, pp, ff = sc['decomp']
    parts = [sc['edge'], np.c_[uu, pp, pp * ff, uu * ff], np.array([H[a] - H[b] for a, b in it['pairs']])]
    hcols = [np.array([H[v] for v in ns])]
    for c in range(sc['all'].shape[1]):
        zc, G = ctx_feats(it, sc['all'][:, c])
        parts.append(zc); hcols.append(G[:, None])
    return np.concatenate(parts, 1), np.concatenate(hcols + [np.ones((len(ns), 1))], 1)


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

    # 3. CRF inputs
    df = item_df(items)
    E = [enumerate_item(it) for it in items]
    ZH = [crf_inputs(it, oof[i], node_feats(it, df, True)) for i, it in enumerate(items)]
    ET = [enumerate_item(it) for it in titems]
    ZHT = [crf_inputs(it, tsc[i], node_feats(it, df, False)) for i, it in enumerate(titems)]

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

    # 4. final CRF on all training fragments -> test marginals
    x = crf_fit(Batch(E, [z for z, h in ZH], [h for z, h in ZH]), best[1])
    tm = crf_marginals(x, Batch(ET, [z for z, h in ZHT], [h for z, h in ZHT]))
    sub = write_submission(te, titems, tm, out_path)
    print(f'wrote {len(sub)} rows to {out_path}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
