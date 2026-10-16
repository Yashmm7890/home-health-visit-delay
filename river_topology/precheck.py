"""Static + CSV precheck for the river topology submission.

usage: python3 precheck.py solution.py --csv submission.csv --sample sample_submission.csv --test test.csv
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
    if re.search(r'\bS0[0-7]\b', src):
        for m in re.finditer(r"['\"]S0[0-7]['\"]", src):
            line = src[:m.start()].count('\n') + 1
            print('FLAG hard-coded slot literal line %d' % line); bad += 1
    return bad


def parse_ok(prog, slots):
    try:
        toks = prog.split(' ')
        if prog != prog.strip() or any(t == '' for t in toks):
            return False
        child_seen, par_seen = set(), set()
        for t in toks:
            ps, c = t.split('>')
            ps = ps.split('+')
            if ps != sorted(ps) or len(set(ps)) != len(ps) or c in ps:
                return False
            if c in child_seen or c not in slots:
                return False
            child_seen.add(c)
            for p in ps:
                if p in par_seen or p not in slots:
                    return False
                par_seen.add(p)
        return len(par_seen) == len(slots) - 1
    except ValueError:
        return False


def check_csv(csv, sample, test):
    sub = pd.read_csv(csv)
    smp = pd.read_csv(sample)
    bad = 0
    if list(sub.columns) != list(smp.columns):
        print('FAIL columns', list(sub.columns)); bad += 1
    if sub['id'].duplicated().any():
        print('FAIL duplicate ids'); bad += 1
    if set(sub['id']) != set(smp['id']):
        print('FAIL id set mismatch'); bad += 1
    if sub.isna().any().any():
        print('FAIL NaN values'); bad += 1
    if test:
        te = pd.read_csv(test)
        cc = dict(zip(te['id'], te['channel_count']))
        nbad = 0
        for i, p in zip(sub['id'], sub['relay_program']):
            slots = set('S%02d' % k for k in range(cc[i]))
            nbad += not parse_ok(str(p), slots)
        if nbad:
            print('FAIL %d malformed programs' % nbad); bad += 1
    print('csv rows', len(sub), 'unique programs', sub['relay_program'].nunique())
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('solution')
    ap.add_argument('--csv')
    ap.add_argument('--sample')
    ap.add_argument('--test')
    a = ap.parse_args()
    nb = check_source(a.solution)
    if a.csv:
        nb += check_csv(a.csv, a.sample, a.test)
    print('PRECHECK', 'FAIL' if nb else 'PASS', nb)
    sys.exit(1 if nb else 0)
