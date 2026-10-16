"""Static + CSV precheck for ShardFit submissions.

usage: python3 precheck.py solution.py --csv submission.csv --sample sample_submission.csv
"""
import argparse
import re
import sys

import pandas as pd

BANNED = [
    (r'\btime\.time\(|\btime\.monotonic\(|perf_counter|datetime\.now\(', 'clock read'),
    (r'os\.environ(?!\[.(OMP|MKL|OPENBLAS)_NUM_THREADS.\]\s*=)', 'environment read'),
    (r'cpu_count\(', 'host cpu_count'),
    (r'requests\.|urllib|http\.client|socket\.', 'network'),
    (r'from_pretrained|torch\.hub|timm\.create_model', 'pretrained'),
    (r'except\s*(Exception)?\s*:', 'broad except (model fallback?)'),
]
TOK = re.compile(r'^c(0\d|1[01]):R[012]$')


def check_source(path):
    src = open(path, encoding='utf-8').read()
    bad = 0
    for pat, why in BANNED:
        for m in re.finditer(pat, src):
            line = src[:m.start()].count('\n') + 1
            print('FLAG %-28s line %d: %s' % (why, line, src.splitlines()[line - 1].strip()[:100]))
            bad += 1
    if len(src.encode('utf-8')) > 512000:
        print('FAIL source exceeds 512000 bytes'); bad += 1
    return bad


def prog_ok(p):
    if p == 'NONE':
        return True
    toks = p.split(' ')
    if not (1 <= len(toks) <= 6) or any(not TOK.match(t) for t in toks):
        return False
    hs = [t.split(':')[0] for t in toks]
    return hs == sorted(hs) and len(set(hs)) == len(hs)


def check_csv(csv, sample):
    sub = pd.read_csv(csv, dtype=str, keep_default_na=False)
    smp = pd.read_csv(sample, dtype=str)
    bad = 0
    if list(sub.columns) != ['id', 'insertion_program']:
        print('FAIL columns', list(sub.columns)); bad += 1
    if len(sub) != len(smp):
        print('FAIL row count', len(sub), len(smp)); bad += 1
    if sub['id'].duplicated().any():
        print('FAIL duplicate ids'); bad += 1
    if set(sub['id']) != set(smp['id']):
        print('FAIL id set mismatch'); bad += 1
    if (sub['insertion_program'] == '').any() or pd.read_csv(csv).isna().any().any():
        print('FAIL empty / NaN values'); bad += 1
    nbad = sum(not prog_ok(p) for p in sub['insertion_program'])
    if nbad:
        print('FAIL %d malformed programs' % nbad); bad += 1
    print('csv rows', len(sub), 'mean tokens %.3f' % sub['insertion_program'].str.split().str.len().mean())
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('solution')
    ap.add_argument('--csv')
    ap.add_argument('--sample')
    a = ap.parse_args()
    nb = check_source(a.solution)
    if a.csv:
        nb += check_csv(a.csv, a.sample)
    print('PRECHECK', 'FAIL' if nb else 'PASS', nb)
    sys.exit(1 if nb else 0)
