"""
WS3 / R2.1 preliminary: model cache check, pre-stage, and test drive on real text.

Two ways to run it.

1. Pre-stage the encoders into a cache (run anywhere with network access; from
   prospero the cm cache is writable over sshfs):

     python 10b_model_check.py --prestage \
       --hf_home /mnt/sciclone/proj-ds/geograd/stmorse/.cache/huggingface

2. Test drive in a cm pod (no network needed once pre-staged). Run from src/ in
   the pod checkout, with HF_HOME set by the manifest:

     python 10b_model_check.py --seed 0

What the test drive checks, all on real dump text:
  a. each model's module list, max_seq_length, dim, and whether its output is
     already unit-norm (we L2-normalize regardless, but we want it on record);
  b. that the ws3_common stream filter reproduces the stored row count for a
     small month (vs totals.csv and vs labels);
  c. **ID check** — streamed comment id == metadata id, for sampled rows of a
     month that has metadata (exact, 2006-01 … 2015-10);
  d. **cosine check** — MiniLM re-embedded from retrieved text vs the stored
     embedding rows, for a zarr month (the only option for 2015-11 onward);
  e. encode throughput per model at the pod's thread count, which sizes the
     retrieval Job;
  f. token lengths per tokenizer, since the model-B encoders have larger
     max_seq_length than MiniLM's 256.

Outputs JSON to --out_path (default ../output/ws3/10b/, gitignored).
"""

import argparse
import configparser
import json
import os
import time

import numpy as np

DEFAULT_MODELS = [
    'sentence-transformers/all-MiniLM-L6-v2',    # model A, the published encoder
    'sentence-transformers/all-mpnet-base-v2',   # model B1, same recipe
    'BAAI/bge-base-en-v1.5',                     # model B2, different recipe
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--models', type=str, default=','.join(DEFAULT_MODELS))
    p.add_argument('--prestage', action='store_true',
                   help='allow downloads; otherwise the run is offline')
    p.add_argument('--hf_home', type=str, default=None,
                   help='overrides $HF_HOME / config hf_path')
    p.add_argument('--threads', type=int, default=0,
                   help='torch CPU threads; 0 = leave alone. In k8s set this to '
                        'the pod cpu limit (os.cpu_count sees the whole node).')
    p.add_argument('--month', type=str, default='2006-06',
                   help='small month, streamed in full: filter + ID + cosine checks')
    p.add_argument('--late_month', type=str, default='2016-03',
                   help='no-metadata month, head of stream only: cosine check')
    p.add_argument('--late_limit', type=int, default=200000,
                   help='rows to stream from the late month (0 = skip)')
    p.add_argument('--n_check', type=int, default=300,
                   help='rows per verification check')
    p.add_argument('--n_bench', type=int, default=4000,
                   help='comments per throughput benchmark (0 = skip)')
    p.add_argument('--run_subpath', type=str, default='mbkm_50')
    p.add_argument('--models_only', action='store_true',
                   help='skip every data check; use to pre-stage models when the '
                        'dumps are not reachable (e.g. sshfs down)')
    p.add_argument('--out_path', type=str, default='../output/ws3/10b')
    return p.parse_args()


def model_report(name, offline, bench_texts, n_check_texts, t0):
    """Load a model, describe it, encode a little real text."""
    from sentence_transformers import SentenceTransformer
    import torch

    out = {'name': name, 'cached': None}
    kw = {'local_files_only': True} if offline else {}
    try:
        m = SentenceTransformer(name, device='cpu', **kw)
        out['cached'] = True
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        out['cached'] = False
        out['error'] = f'{type(exc).__name__}: {exc}'[:300]
        print(f'  ! {name}: not available offline ({type(exc).__name__})')
        return out, None

    out['modules'] = [type(mod).__name__ for mod in m]
    out['ends_in_normalize'] = out['modules'][-1] == 'Normalize'
    out['max_seq_length'] = int(m.max_seq_length)
    out['dim'] = int(m.get_sentence_embedding_dimension())
    out['tokenizer_model_max_length'] = int(m.tokenizer.model_max_length)

    # output norms, before any normalization of ours
    E = m.encode(n_check_texts[:32], convert_to_numpy=True, show_progress_bar=False)
    nrm = np.linalg.norm(E.astype(np.float64), axis=1)
    out['output_norm'] = {'min': float(nrm.min()), 'max': float(nrm.max())}

    # token lengths on real bodies (bodies are already 2000-char truncated)
    tl = [len(m.tokenizer(t)['input_ids']) for t in n_check_texts[:2000]]
    tl = np.array(tl)
    out['token_len'] = {
        'median': float(np.median(tl)),
        'p99': float(np.percentile(tl, 99)),
        'frac_over_max_seq': float((tl > m.max_seq_length).mean()),
        'frac_over_256': float((tl > 256).mean()),
    }

    if bench_texts:
        for bs in (64,):
            t = time.time()
            m.encode(bench_texts, batch_size=bs, convert_to_numpy=True,
                     show_progress_bar=False)
            dt = time.time() - t
            out['throughput_sent_per_s'] = round(len(bench_texts) / dt, 1)
            out['bench_n'] = len(bench_texts)
            out['bench_batch_size'] = bs
            out['torch_threads'] = torch.get_num_threads()

    print(f'  {name}: dim {out["dim"]}, max_seq {out["max_seq_length"]}, '
          f'modules {"->".join(out["modules"])}, '
          f'norm [{out["output_norm"]["min"]:.4f}, {out["output_norm"]["max"]:.4f}], '
          f'{out.get("throughput_sent_per_s", "-")} sent/s  ({time.time()-t0:.1f}s)')
    return out, m


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    # HF_HOME must be set before sentence_transformers / huggingface_hub import.
    hf = args.hf_home or os.environ.get('HF_HOME') or g['hf_path']
    os.environ['HF_HOME'] = hf
    if not args.prestage:
        os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    import ws3_common as wc

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_path, exist_ok=True)
    t0 = time.time()
    res = {
        'params': vars(args),
        'hf_home': hf,
        'offline': not args.prestage,
        'cpu_count_visible': os.cpu_count(),
        'torch_threads': torch.get_num_threads(),
        'paths': {k: g[k] for k in ('data_path', 'embed_path', 'save_path')},
    }
    print(f'HF_HOME {hf}   offline={not args.prestage}   '
          f'torch threads {torch.get_num_threads()} of {os.cpu_count()} visible')

    if args.models_only:
        print('\nModels only (no data checks):')
        sample = ['a short reddit comment about politics',
                  'this is a longer comment ' * 40]
        res['models'] = [model_report(n.strip(), not args.prestage, None,
                                      sample * 16, t0)[0]
                         for n in args.models.split(',')]
        with open(os.path.join(args.out_path, 'results_models_only.json'), 'w') as f:
            json.dump(res, f, indent=2, default=str)
        print(f'\nWrote {args.out_path}/results_models_only.json  '
              f'({time.time()-t0:.1f}s)')
        return

    # ---- 1. stream a small month in full; check the filter ----
    ym = args.month
    dump = wc.find_dump(g['data_path'], ym)
    print(f'\nStreaming {ym} in full ({os.path.getsize(dump)/1e6:.1f} MB) ...')
    t = time.time()
    rows, bodies, ids = [], [], []
    for i, body, cid in wc.stream_rows(dump, fields=('id',)):
        rows.append(i)
        bodies.append(body)
        ids.append(cid)
    dt = time.time() - t
    totals = wc.load_totals(g['save_path'])
    labels = wc.load_labels(g['save_path'], ym, args.run_subpath)
    res['filter_check'] = {
        'month': ym, 'dump': dump,
        'kept_rows': len(rows),
        'totals_csv': totals.get(ym),
        'labels_len': int(len(labels)),
        'match_totals': len(rows) == totals.get(ym),
        'match_labels': len(rows) == len(labels),
        'contiguous_index': rows == list(range(len(rows))),
        'stream_sec': round(dt, 1),
        'rows_per_sec': round(len(rows) / dt, 0),
    }
    print(f'  kept {len(rows):,} | totals.csv {totals.get(ym):,} | labels {len(labels):,} '
          f'| match={res["filter_check"]["match_totals"] and res["filter_check"]["match_labels"]} '
          f'| {res["filter_check"]["rows_per_sec"]:,.0f} rows/s')

    # ---- 2. ID check against metadata (exact, months <= 2015-10) ----
    if wc.has_metadata(ym):
        pick = rng.choice(len(rows), size=min(args.n_check, len(rows)), replace=False)
        want = sorted(int(rows[i]) for i in pick)
        meta = wc.load_metadata_ids(g['save_path'], ym, rows=want)
        agree = sum(1 for i in want if meta.get(i) == ids[i])
        res['id_check'] = {
            'month': ym, 'n': len(want), 'found_in_metadata': len(meta),
            'agree': agree, 'all_agree': agree == len(want),
        }
        print(f'  ID check: {agree}/{len(want)} streamed ids match metadata')
    else:
        res['id_check'] = {'month': ym, 'skipped': 'no metadata'}

    # ---- 3. cosine check vs stored embeddings ----
    from sentence_transformers import SentenceTransformer
    try:
        mini = SentenceTransformer(DEFAULT_MODELS[0], device='cpu',
                                   local_files_only=not args.prestage)
    except Exception as exc:                       # noqa: BLE001
        mini = None
        res['cosine_check'] = {'error': f'MiniLM unavailable: {type(exc).__name__}'}
        print(f'  ! MiniLM unavailable, skipping cosine checks ({exc})'[:200])

    def cosine_check(ym_, rows_, bodies_):
        pick = rng.choice(len(rows_), size=min(args.n_check, len(rows_)), replace=False)
        idx = np.array([rows_[i] for i in pick])
        txt = [bodies_[i] for i in pick]
        E = wc.l2_normalize(mini.encode(txt, convert_to_numpy=True,
                                        show_progress_bar=False))
        S = wc.l2_normalize(wc.read_embedding_rows(g['embed_path'], ym_, idx))
        cos = np.sum(E * S, axis=1)
        kind, _ = wc.embedding_store(g['embed_path'], ym_)
        out = {
            'month': ym_, 'store': kind, 'n': int(len(cos)),
            'cos_min': float(cos.min()), 'cos_median': float(np.median(cos)),
            'frac_below_0.99': float((cos < 0.99).mean()),
        }
        print(f'  cosine check {ym_} ({kind}): median {out["cos_median"]:.5f}, '
              f'min {out["cos_min"]:.5f}, frac<0.99 {out["frac_below_0.99"]:.4f}')
        return out

    if mini is not None:
        res['cosine_check'] = cosine_check(ym, rows, bodies)

        # late month: head of stream only (no metadata there, so cosine is the
        # only check available; the full Job checks every month this way)
        if args.late_limit:
            lm = args.late_month
            dump2 = wc.find_dump(g['data_path'], lm)
            print(f'\nStreaming head of {lm} ({args.late_limit:,} rows of '
                  f'{os.path.getsize(dump2)/1e9:.1f} GB) ...')
            t = time.time()
            r2, b2 = [], []
            for i, body in wc.stream_rows(dump2, limit=args.late_limit):
                r2.append(i)
                b2.append(body)
            res['late_stream'] = {
                'month': lm, 'rows': len(r2), 'sec': round(time.time() - t, 1),
                'rows_per_sec': round(len(r2) / (time.time() - t), 0),
                'has_metadata': wc.has_metadata(lm),
            }
            print(f'  {len(r2):,} rows in {res["late_stream"]["sec"]}s '
                  f'({res["late_stream"]["rows_per_sec"]:,.0f} rows/s)')
            res['cosine_check_late'] = cosine_check(lm, r2, b2)

    # ---- 4. per-model report + throughput ----
    print('\nModels:')
    bench = None
    if args.n_bench:
        bench = [bodies[i] for i in
                 rng.choice(len(bodies), size=min(args.n_bench, len(bodies)),
                            replace=False)]
    res['models'] = []
    for name in args.models.split(','):
        rep, _ = model_report(name.strip(), not args.prestage, bench, bodies, t0)
        res['models'].append(rep)

    # ---- projection for the retrieval Job ----
    tp = {m['name']: m.get('throughput_sent_per_s') for m in res['models']
          if m.get('throughput_sent_per_s')}
    if tp:
        per_month = 250000
        res['job_projection'] = {
            'sample_per_month': per_month,
            'embed_minutes_per_month': {k: round(per_month / v / 60, 1)
                                        for k, v in tp.items()},
            'note': 'add the dump stream time; see late_stream rows_per_sec',
        }
        print(f'\nProjected embed time per month ({per_month:,} comments): '
              + ', '.join(f'{k.split("/")[-1]} {v} min'
                          for k, v in res['job_projection']
                          ['embed_minutes_per_month'].items()))

    with open(os.path.join(args.out_path, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f'\nWrote {args.out_path}/results.json  ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
