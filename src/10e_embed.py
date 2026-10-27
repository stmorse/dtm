"""
WS3 / R2.1 stage 3 of 3: embed the retrieved text with each encoder.

GPU-bound and trivially resumable: one output file per (model, month). Measured
on the 5090 (fp16, real dump text): MiniLM 20.7k, mpnet 6.7k, bge 8.5k
sentences/s, i.e. under a minute per 250k month per model.

MiniLM is re-embedded rather than read from `embed_path` on purpose. A month's
sampled rows are scattered across every zarr chunk, so reading them touches
nearly the whole month (tens of GB), while re-embedding takes seconds. The
2026-10-19 check found median and min cosine = 1.00000 between re-embedded and
stored rows, so the two are interchangeable.

All three encoders end in a Normalize module, so every space is unit-norm; we
L2-normalize again anyway, since fp16 storage perturbs the norm slightly.

--max_seq caps every model's input window (default 256, MiniLM's limit) so the
model-B spaces see exactly what MiniLM saw. Pass 0 to leave each model's own
window (mpnet 384, bge 512).

Run from src/:
    python 10e_embed.py --seed 0
    python 10e_embed.py --seed 0 --months 2006-06,2008-06 --models minilm,mpnet
"""

import argparse
import configparser
import gzip
import json
import os
import time

import numpy as np

import ws3_common as wc

MODELS = {
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'bge': 'BAAI/bge-base-en-v1.5',
}


def read_text(path):
    rows, bodies = [], []
    with gzip.open(path, 'rt') as f:
        for line in f:
            d = json.loads(line)
            rows.append(d['row'])
            bodies.append(d['body'])
    order = np.argsort(np.asarray(rows, dtype=np.int64))
    rows = np.asarray(rows, dtype=np.int64)[order]
    return rows, [bodies[i] for i in order]


def main():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--models', type=str, default='minilm,mpnet,bge')
    p.add_argument('--months', type=str, default='')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--fp16', type=int, default=1)
    p.add_argument('--max_seq', type=int, default=256,
                   help='cap every model to this window; 0 = model default')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--rev_subpath', type=str, default='rev/ws3')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    os.environ['HF_HOME'] = wc.hf_home(g)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from sentence_transformers import SentenceTransformer

    rev = os.path.join(g['save_path'], args.rev_subpath)
    months = ([m.strip() for m in args.months.split(',') if m.strip()]
              or sorted(f[len('text_'):-len('.jsonl.gz')] for f in
                        os.listdir(os.path.join(rev, 'text'))
                        if f.endswith('.jsonl.gz')))
    tags = [t.strip() for t in args.models.split(',') if t.strip()]
    for t in tags:
        if t not in MODELS:
            raise SystemExit(f'unknown model tag {t}; choose from {list(MODELS)}')

    print(f'{len(months)} months x {len(tags)} models on {args.device}'
          f'{" fp16" if args.fp16 else ""}, max_seq {args.max_seq or "model default"}')
    t0 = time.time()
    for tag in tags:
        out_dir = os.path.join(rev, 'emb', tag)
        os.makedirs(out_dir, exist_ok=True)
        kw = {'model_kwargs': {'torch_dtype': 'float16'}} if args.fp16 else {}
        model = SentenceTransformer(MODELS[tag], device=args.device, **kw)
        if args.max_seq:
            model.max_seq_length = args.max_seq
        dim = model.get_sentence_embedding_dimension()
        meta = {'tag': tag, 'hf_name': MODELS[tag], 'dim': dim,
                'max_seq_length': int(model.max_seq_length),
                'modules': [type(m).__name__ for m in model],
                'fp16': bool(args.fp16), 'device': args.device,
                'batch_size': args.batch_size, 'seed': args.seed}
        with open(os.path.join(out_dir, 'model.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'\n[{tag}] {MODELS[tag]} dim {dim} '
              f'max_seq {model.max_seq_length}')

        for ym in months:
            out = os.path.join(out_dir, f'emb_{ym}.npz')
            if os.path.exists(out) and not args.overwrite:
                print(f'  {ym}: exists, skipping')
                continue
            tp = os.path.join(rev, 'text', f'text_{ym}.jsonl.gz')
            if not os.path.exists(tp):
                print(f'  {ym}: no text file, skipping')
                continue
            t = time.time()
            rows, bodies = read_text(tp)
            t_read = time.time() - t
            t = time.time()
            X = model.encode(bodies, batch_size=args.batch_size,
                             convert_to_numpy=True, show_progress_bar=False)
            t_enc = time.time() - t
            X = wc.l2_normalize(X).astype(np.float16)
            # write through a handle: np.savez would append .npz to the name
            tmp = out + '.part'
            with open(tmp, 'wb') as fh:
                np.savez(fh, rows=rows, X=X, meta=json.dumps(meta))
            os.replace(tmp, out)
            print(f'  {ym}: {len(rows):>8,} x {dim} | read {t_read:5.1f}s '
                  f'| encode {t_enc:6.1f}s ({len(rows)/max(t_enc,1e-9):>7,.0f}/s) '
                  f'| {os.path.getsize(out)/1e6:6.1f} MB', flush=True)
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    print(f'\nDone in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
