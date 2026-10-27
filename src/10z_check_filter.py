"""
WS3 / R2.1: assert the fast byte-level keep/skip filter is equivalent to the
exact JSON filter of `1_embed.py`.

Every downstream row index depends on the kept-row count, so this has to hold
for every compression format and era. For each month it compares the two paths
row by row (index and comment id) and, for months streamed in full, checks the
count against totals.csv and labels.

Run from src/:
    python 10z_check_filter.py --months 2006-06,2008-06 --full
    python 10z_check_filter.py --limit 200000          # spans formats and eras
"""

import argparse
import configparser
import os
import time

import numpy as np

import ws3_common as wc

# one per format/era: bz2 early, bz2 mid, bz2 late, xz, zst
DEFAULT = ['2006-06', '2008-06', '2012-06', '2015-11', '2018-06', '2021-06']


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--months', type=str, default=','.join(DEFAULT))
    p.add_argument('--limit', type=int, default=200000,
                   help='rows per month; ignored with --full')
    p.add_argument('--full', action='store_true',
                   help='stream each month in full (only for small months)')
    args = p.parse_args()

    totals = wc.load_totals(g['save_path'])
    lim = None if args.full else args.limit
    fails = []
    for ym in [m.strip() for m in args.months.split(',') if m.strip()]:
        dump = wc.find_dump(g['data_path'], ym)
        ext = dump.rsplit('.', 1)[-1]
        t = time.time()
        fast = list(wc.stream_rows(dump, limit=lim, fields=('id',), strict=False))
        t_fast = time.time() - t
        t = time.time()
        slow = list(wc.stream_rows(dump, limit=lim, fields=('id',), strict=True))
        t_slow = time.time() - t

        same_n = len(fast) == len(slow)
        same_rows = same_n and all(a[0] == b[0] for a, b in zip(fast, slow))
        same_ids = same_n and all(a[2] == b[2] for a, b in zip(fast, slow))
        ok = same_n and same_rows and same_ids
        note = ''
        if args.full:
            n_tot = totals.get(ym)
            n_lab = len(wc.load_labels(g['save_path'], ym))
            match = (len(fast) == n_tot == n_lab)
            ok = ok and match
            note = (f' | totals {n_tot:,} labels {n_lab:,} '
                    f'{"match" if match else "MISMATCH"}')
        print(f'{ym} ({ext:>3}): fast {len(fast):>8,} rows @ '
              f'{len(fast)/t_fast:>7,.0f}/s | strict {len(slow):>8,} @ '
              f'{len(slow)/t_slow:>7,.0f}/s | speedup {t_slow/t_fast:4.1f}x '
              f'| {"EQUIVALENT" if ok else "DIFFERS"}{note}')
        if not ok:
            fails.append(ym)

    print()
    if fails:
        raise SystemExit(f'FILTER MISMATCH in {fails} — do not run retrieval')
    print('filter equivalence OK')


if __name__ == '__main__':
    main()
