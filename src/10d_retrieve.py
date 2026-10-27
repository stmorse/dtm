"""
WS3 / R2.1 stage 2 of 3: retrieve the sampled comment text.

One sequential pass per month over the raw dump, replicating the `1_embed.py`
filter so stream position == embedding row index. Writes gzipped JSONL
(row, id, cluster, body) plus a verification certificate.

A month cannot be split across workers: row indices require counting kept rows
from the start of the file. So parallelism is concurrent months, each
single-threaded, and --workers months run at once.

Verification, per month, without reading any embedding file:
  nearest-centroid check  re-embed a few hundred retrieved bodies with MiniLM
                          and assign each to its nearest published centroid for
                          that month. MBKM labels *are* nearest-centroid
                          assignments, so agreement with labels[row] confirms
                          the row alignment. Misalignment would collapse this to
                          chance (~1/50 = 2%). Available for every month.
  id check                for months <= 2015-10, streamed comment id vs the
                          metadata id for that row index. Exact.

Agreement is not expected to be exactly 1.0: the published embeddings were fp16
on GPU and points near a cell boundary can flip. The metadata months calibrate
that rate, which is then the reference for the months with no metadata.

Run from src/ (long runs: use tmux):
    python 10d_retrieve.py --seed 0 --workers 12
    python 10d_retrieve.py --seed 0 --months 2006-06,2008-06 --workers 2
"""

import argparse
import configparser
import gzip
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import ws3_common as wc

_MINI = None       # per-worker MiniLM, loaded lazily


def get_minilm(hf, offline=True):
    global _MINI
    if _MINI is None:
        os.environ['HF_HOME'] = hf
        if offline:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
        import torch
        torch.set_num_threads(1)         # one month per worker; don't oversubscribe
        from sentence_transformers import SentenceTransformer
        _MINI = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                    device='cpu')
    return _MINI


def retrieve_month(job):
    ym, cfg = job
    t0 = time.time()
    g, a = cfg['g'], cfg['a']
    rev = os.path.join(g['save_path'], a['rev_subpath'])
    text_p = os.path.join(rev, 'text', f'text_{ym}.jsonl.gz')
    cert_p = os.path.join(rev, 'cert', f'cert_{ym}.json')

    if os.path.exists(cert_p) and not a['overwrite'] and not a['verify_only']:
        return {'month': ym, 'status': 'skipped'}
    if a['verify_only'] and not os.path.exists(text_p):
        return {'month': ym, 'status': 'no text to verify'}

    smp = np.load(os.path.join(rev, 'sample', f'sample_{ym}.npz'),
                  allow_pickle=False)
    rows, cluster = smp['rows'], smp['cluster']
    want = set(int(r) for r in rows)
    row_to_cluster = dict(zip(rows.tolist(), cluster.tolist()))

    dump = wc.find_dump(g['data_path'], ym)
    got_rows, got_ids = [], []
    if a['verify_only']:
        # re-verify from the text already on disk; no dump pass, no rewrite
        with gzip.open(text_p, 'rt') as f:
            for line in f:
                d = json.loads(line)
                got_rows.append(d['row'])
                got_ids.append(d['id'])
        n_written = len(got_rows)
        prev = {}
        if os.path.exists(cert_p):
            with open(cert_p) as f:
                prev = json.load(f)
    else:
        tmp = text_p + '.part'
        n_written = 0
        with gzip.open(tmp, 'wt', compresslevel=4) as f:
            for i, body, cid in wc.stream_rows(dump, keep=want,
                                               limit=a['limit_rows'] or None,
                                               fields=('id',)):
                f.write(json.dumps({'row': i, 'id': cid,
                                    'cluster': row_to_cluster[i],
                                    'body': body}) + '\n')
                got_rows.append(i)
                got_ids.append(cid)
                n_written += 1
        os.replace(tmp, text_p)
        prev = {}

    partial = bool(prev.get('partial_limit_rows') if a['verify_only']
                   else a['limit_rows'])
    cert = {
        'month': ym, 'dump': os.path.basename(dump),
        'dump_bytes': os.path.getsize(dump),
        'n_sampled': int(len(rows)), 'n_retrieved': n_written,
        'complete': (n_written == len(rows)) and not partial,
        'partial_limit_rows': (prev.get('partial_limit_rows')
                               if a['verify_only'] else a['limit_rows'] or None),
        'sec': prev.get('sec', round(time.time() - t0, 1)) if a['verify_only']
               else round(time.time() - t0, 1),
        'verify_only_rerun': bool(a['verify_only']),
        'rows_per_sec': None,
        'text_bytes': os.path.getsize(text_p),
    }

    # ---- nearest-centroid check (every month) ----
    rng = np.random.default_rng([a['seed'], int(ym[:4]), int(ym[5:])])
    if n_written:
        k = min(a['verify_n'], n_written)
        pick = rng.choice(n_written, size=k, replace=False)
        sel_rows = [got_rows[i] for i in pick]
        bodies = []
        with gzip.open(text_p, 'rt') as f:
            keep = set(sel_rows)
            for line in f:
                d = json.loads(line)
                if d['row'] in keep:
                    bodies.append((d['row'], d['body'], d['id']))
        mini = get_minilm(cfg['hf'])
        E = wc.l2_normalize(mini.encode([b for _, b, _ in bodies],
                                        convert_to_numpy=True,
                                        show_progress_bar=False))
        cc = wc.load_centroids_month(g['save_path'], ym, a['run_subpath'])
        d2 = ((E[:, None, :] - cc[None, :, :]) ** 2).sum(-1)
        pred = d2.argmin(1)
        truth = np.array([row_to_cluster[r] for r, _, _ in bodies])
        cert['nearest_centroid_check'] = {
            'n': int(len(pred)),
            'agree': float((pred == truth).mean()),
            'chance': round(1.0 / cc.shape[0], 4),
        }

        # ---- id check (months with metadata) ----
        if wc.has_metadata(ym):
            sel = [(r, cid) for r, _, cid in bodies]
            try:
                meta = wc.load_metadata_ids(
                    g['save_path'], ym, rows=[r for r, _ in sel],
                    max_bytes=int(a['metadata_max_gb'] * 1e9))
                agree = sum(1 for r, cid in sel if meta.get(r) == cid)
                cert['id_check'] = {'n': len(sel), 'found': len(meta),
                                    'agree': agree,
                                    'all_agree': agree == len(sel)}
            except wc.MetadataTooLarge as exc:
                # the nearest-centroid check covers this month; rerun later with
                # --verify_only --workers 1 --metadata_max_gb 4 to add the id check
                cert['id_check'] = {'skipped': 'metadata2 too large',
                                    'detail': str(exc)}
            except FileNotFoundError as exc:
                cert['id_check'] = {'error': str(exc)}
        else:
            cert['id_check'] = {'skipped': 'no metadata'}

    os.makedirs(os.path.dirname(cert_p), exist_ok=True)
    with open(cert_p, 'w') as f:
        json.dump(cert, f, indent=2)
    nc = cert.get('nearest_centroid_check', {})
    print(f'{ym}: {n_written:>8,}/{len(rows):,} rows in {cert["sec"]:>7.1f}s '
          f'| {cert["text_bytes"]/1e6:6.1f} MB | nearest-centroid '
          f'{nc.get("agree", float("nan")):.4f} '
          f'| id {cert.get("id_check", {}).get("agree", "-")}', flush=True)
    return cert


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--months', type=str, default='')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--verify_n', type=int, default=400)
    p.add_argument('--limit_rows', type=int, default=0,
                   help='debug: stop each month after this many kept rows; '
                        'marks the certificate partial')
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--metadata_max_gb', type=float, default=0.5,
                   help='skip the id check when the metadata2 pickle exceeds '
                        'this size on disk. It unpickles to roughly 6x the file '
                        'size (a 2.3 GB file -> ~14 GB of Python objects), and '
                        'one per worker OOM-killed the box on 2026-10-20. The '
                        'nearest-centroid check still covers those months.')
    p.add_argument('--verify_only', action='store_true',
                   help='re-run the checks from the text already retrieved and '
                        'rewrite the certificate; no dump pass. Use with '
                        '--workers 1 --metadata_max_gb 4 to add id checks for '
                        'the big metadata2 months one at a time.')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    rev = os.path.join(g['save_path'], args.rev_subpath)
    for sub in ('text', 'cert'):
        os.makedirs(os.path.join(rev, sub), exist_ok=True)
    months = ([m.strip() for m in args.months.split(',') if m.strip()]
              or wc.ym_range('2006-01', '2022-12'))
    missing = [m for m in months
               if not os.path.exists(os.path.join(rev, 'sample',
                                                  f'sample_{m}.npz'))]
    if missing:
        raise SystemExit(f'No sample file for {len(missing)} months '
                         f'(first: {missing[:3]}). Run 10c_sample.py first.')

    cfg = {'g': dict(g), 'a': vars(args), 'hf': wc.hf_home(g)}
    # biggest months first: with a fixed worker count that minimises the tail
    sizes = {m: os.path.getsize(wc.find_dump(g['data_path'], m)) for m in months}
    months = sorted(months, key=lambda m: -sizes[m])
    print(f'{len(months)} months, {sum(sizes.values())/1e12:.3f} TB of dumps, '
          f'{args.workers} workers')

    t0 = time.time()
    jobs = [(m, cfg) for m in months]
    with Pool(args.workers) as pool:
        certs = pool.map(retrieve_month, jobs, chunksize=1)

    done = [c for c in certs if c.get('status') != 'skipped']
    bad = [c for c in done if not c.get('complete')]
    print(f'\n{len(done)} months retrieved, {len(certs)-len(done)} skipped, '
          f'{len(bad)} incomplete, in {(time.time()-t0)/60:.1f} min')
    if done:
        agree = [c['nearest_centroid_check']['agree'] for c in done
                 if 'nearest_centroid_check' in c]
        if agree:
            print(f'nearest-centroid agreement: min {min(agree):.4f}, '
                  f'median {float(np.median(agree)):.4f}')
        ids = [c['id_check'] for c in done if c.get('id_check', {}).get('n')]
        if ids:
            print(f'id check: {sum(i["agree"] for i in ids)}/'
                  f'{sum(i["n"] for i in ids)} across {len(ids)} months')
    if bad:
        print('INCOMPLETE: ' + ', '.join(c['month'] for c in bad[:10]))


if __name__ == '__main__':
    main()
