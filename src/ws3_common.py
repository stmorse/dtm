"""
WS3 shared helpers (R2.1, R2.3, R2.8).

Everything that touches the raw dumps or the stored embeddings lives here, so the
WS3 scripts (10*_*.py) share one implementation of the `1_embed.py` filter. That
filter defines the row index of every embedding, so it must stay byte-identical
to `1_embed.py` / `utils.read_sentences`:

    skip entries with no 'body' or author == '[deleted]', truncate body at 2000
    characters, and increment the row counter only for kept entries.

Nothing here imports from the pipeline scripts (their module names start with a
digit, so they are not importable) and nothing here modifies them.
"""

import bz2
import gzip
import json
import lzma
import os
import pickle

import numpy as np
import zstandard as zstd

MAX_CHARS = 2000        # 1_embed.py body truncation
K_CLUSTERS = 50         # mbkm_50


# ----- raw dumps -----

RAW_BUFFER = 8 * 1024 * 1024    # see open_compressed
STREAM_BLOCK = 4 * 1024 * 1024  # decompressed bytes per read in stream_rows


def open_compressed(file_path, raw_buffer=RAW_BUFFER):
    """
    Same formats as utils.open_compressed, but the *underlying* file gets a
    large buffer. The pipeline version lets each decompressor read the raw file
    in io.DEFAULT_BUFFER_SIZE (8 KiB) chunks, which is fine on cm's NFS but
    latency-bound over sshfs, where a full-corpus pass is 1.4 TB. Decompression
    is unaffected; only the read size changes.
    """
    raw = open(file_path, 'rb', buffering=raw_buffer)
    if file_path.endswith('.bz2'):
        return bz2.BZ2File(raw, 'rb')
    elif file_path.endswith('.xz'):
        return lzma.LZMAFile(raw, 'rb')
    elif file_path.endswith('.zst'):
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        return dctx.stream_reader(raw)
    raise ValueError(f'Unsupported file extension: {file_path}')


def find_dump(data_path, ym):
    """Path to RC_<ym>.{bz2,xz,zst}."""
    for ext in ['.bz2', '.xz', '.zst']:
        p = os.path.join(data_path, f'RC_{ym}{ext}')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'No dump for {ym} in {data_path}')


B_BODY = b'"body"'
B_DEL_AUTHOR = b'"author":"[deleted]"'
B_DEL_AUTHOR_SP = b'"author": "[deleted]"'


def iter_lines(f, block=STREAM_BLOCK):
    """
    Lines from a decompressing stream, reading in large blocks.

    Not `line, buf = buf.split(b'\\n', 1)` in a loop: that copies the whole
    remaining buffer per line, which is quadratic in the block size (measured
    94k -> 13k lines/s at a 4 MiB block). Splitting the block once and carrying
    the partial tail is linear.
    """
    carry = b''
    for chunk in iter(lambda: f.read(block), b''):
        lines = (carry + chunk).split(b'\n')
        carry = lines.pop()
        for ln in lines:
            yield ln
    if carry:
        yield carry


def stream_rows(file_path, keep=None, limit=None, fields=(), strict=False,
                block=STREAM_BLOCK):
    """
    Yield (row_idx, body[, *fields]) for kept entries, replicating the
    `1_embed.py` filter. `row_idx` is the embedding row index.

    keep   : optional set of row indices to yield (others are counted but
             skipped). Stops once the largest wanted index is past.
    limit  : stop after this many kept rows (ignoring `keep`).
    fields : extra JSON keys to return with the body, e.g. ('id',).
    strict : parse every line as JSON to decide keep/skip, exactly as
             `1_embed.py` does. The default fast path makes that decision on
             raw bytes and parses JSON only for rows it yields, which is ~8x
             faster on a full pass (63k vs 8k rows/s) because the discarded
             99.5% are never parsed.

    The two paths must agree on the kept-row count, or every downstream row
    index is wrong. `10z_check_filter.py` asserts that on real months, and a
    full-month run also checks the count against totals.csv.
    """
    stop_after = max(keep) if keep else None
    i = 0
    with open_compressed(file_path) as f:
        for ln in iter_lines(f, block):
            if not ln.strip():
                continue
            if strict:
                entry = json.loads(ln.decode('utf-8'))
                if 'body' not in entry or entry['author'] == '[deleted]':
                    continue
            else:
                # cheap keep/skip on raw bytes. Pushshift writes compact JSON,
                # but accept a space after the key too; a body containing this
                # text would have escaped quotes, so it cannot false-positive.
                if B_BODY not in ln:
                    continue
                if B_DEL_AUTHOR in ln or B_DEL_AUTHOR_SP in ln:
                    continue
                entry = None
            if keep is None or i in keep:
                if entry is None:
                    entry = json.loads(ln.decode('utf-8'))
                body = entry['body']
                if len(body) > MAX_CHARS:
                    body = body[:MAX_CHARS]
                yield (i, body) + tuple(entry.get(k) for k in fields)
            i += 1
            if limit is not None and i >= limit:
                return
            if stop_after is not None and i > stop_after:
                return


# ----- stored embeddings -----

def embedding_store(embed_path, ym):
    """('zarr'|'npz', path). Prefers zarr: npz months can be tens of GB and
    np.load decompresses the whole array."""
    z = os.path.join(embed_path, f'embeddings_{ym}.zarr')
    if os.path.exists(z):
        return 'zarr', z
    n = os.path.join(embed_path, f'embeddings_{ym}.npz')
    if os.path.exists(n):
        return 'npz', n
    raise FileNotFoundError(f'No embeddings for {ym} in {embed_path}')


def read_embedding_rows(embed_path, ym, rows):
    """
    Rows of a month's stored embeddings, in the order given.

    zarr: reads only the chunks that hold the wanted rows.
    npz : loads the whole array (use sparingly; see embedding_store).
    """
    rows = np.asarray(rows, dtype=np.int64)
    kind, path = embedding_store(embed_path, ym)
    if kind == 'npz':
        X = np.load(path)['embeddings']
        return np.asarray(X[rows])

    import zarr
    z = zarr.open(path, mode='r')
    chunk = z.chunks[0]
    out = np.empty((len(rows), z.shape[1]), dtype=z.dtype)
    order = np.argsort(rows)
    srt = rows[order]
    for c in np.unique(srt // chunk):
        lo, hi = c * chunk, min((c + 1) * chunk, z.shape[0])
        sel = (srt >= lo) & (srt < hi)
        block = np.asarray(z[lo:hi])
        out[order[sel]] = block[srt[sel] - lo]
    return out


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


# ----- labels, counts, metadata -----

def load_labels(save_path, ym, run_subpath='mbkm_50'):
    p = os.path.join(save_path, run_subpath, 'labels', f'labels_{ym}.npz')
    return np.load(p)['labels']


def load_centroids_month(save_path, ym, run_subpath='mbkm_50'):
    p = os.path.join(save_path, run_subpath, 'models', f'model_cc_{ym}.npz')
    return np.load(p)['cc']


def load_totals(save_path):
    """{ 'YYYY-MM': n_rows } from totals.csv."""
    import csv
    out = {}
    with open(os.path.join(save_path, 'totals.csv')) as f:
        for row in csv.DictReader(f):
            out[f"{row['year']}-{int(row['month']):02d}"] = int(row['subtotal'])
    return out


def has_metadata(ym):
    """True if a (row_idx -> comment id) map exists for this month."""
    return '2006-01' <= ym <= '2015-10'


class MetadataTooLarge(Exception):
    """metadata2 pickle too big to load within the caller's memory budget."""


def load_metadata_ids(save_path, ym, rows=None, max_bytes=None):
    """
    { row_idx: comment_id } for this month, from metadata/ (gzip CSV despite the
    .csv name; 2006-01 … 2012-12) or metadata2/ (pickled lists; 2007-01 …
    2015-10). `rows` restricts the result.

    The two sources differ enormously in cost. The gzip CSV is *streamed*, so
    memory is O(len(rows)). The pickle is a list of ~n 4-element Python lists
    and `pickle.load` must materialise all of it: a 2.3 GB file for 2015-10
    expands to 10-15 GB of objects. Loading one per worker OOM-killed a 62 GB
    machine on 2026-10-20. `max_bytes` therefore guards the pickle path and
    raises MetadataTooLarge rather than trying; callers fall back to the
    nearest-centroid check, which is stronger anyway and needs no metadata.
    """
    import csv
    want = None if rows is None else set(int(r) for r in rows)
    csv_p = os.path.join(save_path, 'metadata', f'metadata_{ym}.csv')
    pkl_p = os.path.join(save_path, 'metadata2', f'metadata_{ym}.pkl')
    out = {}
    if os.path.exists(csv_p):
        with gzip.open(csv_p, 'rt', newline='') as f:
            for row in csv.DictReader(f):
                i = int(row['idx'])
                if want is None or i in want:
                    out[i] = row['id']
        return out
    if os.path.exists(pkl_p):
        size = os.path.getsize(pkl_p)
        if max_bytes is not None and size > max_bytes:
            raise MetadataTooLarge(
                f'{ym}: metadata2 pickle is {size/1e9:.2f} GB on disk '
                f'(~{size*6/1e9:.0f} GB once unpickled), over the '
                f'{max_bytes/1e9:.2f} GB budget')
        with open(pkl_p, 'rb') as f:
            recs = pickle.load(f)
        # The row index is the record's POSITION in the list, not rec[0].
        # rec[0] is inconsistent across eras: for 2007-06 it equals the position,
        # but for 2015-07 it is the 1M-row batch counter (max 54 for 54.8M rows),
        # which is what the commented-out NPZ path in 1_embed.py writes
        # (`df.append([k, ...])`, k = batch count). Keying on rec[0] made the
        # id check fail for every row of 2015-07/08/10 while nearest-centroid
        # agreement was >=0.99, i.e. the check was wrong, not the retrieval.
        # Position-based lookup is verified correct in both eras.
        for i, rec in enumerate(recs):         # [batch_or_row, author, id, created_utc]
            if want is None or i in want:
                out[i] = rec[2]
        del recs
        return out
    raise FileNotFoundError(f'No metadata for {ym}')


# ----- misc -----

def ym_range(start='2006-01', end='2022-12'):
    out = []
    y, m = int(start[:4]), int(start[5:])
    ey, em = int(end[:4]), int(end[5:])
    while (y, m) <= (ey, em):
        out.append(f'{y}-{m:02d}')
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def hf_home(config_general):
    """HF_HOME: pod env wins, then config.ini's hf_path. The cm cache is
    /sciclone/proj-ds/geograd/stmorse/.cache/huggingface (see response.md,
    2026-10-19 infrastructure note)."""
    return os.environ.get('HF_HOME') or config_general['hf_path']


# ----- alignment and the drift test -----
# These two reproduce the published alignment and WS1's locked statistic. They
# live here so 10g_compare.py (R2.1 arms B2/A) and 10k_nullrun.py (R2.8) share
# one implementation instead of two that can drift apart. `umap` and `drift` are
# imported lazily, because 10c/10d/10e import this module inside cm pods where
# neither is needed.
#
# NOTE for R2.8: the published run used min_cluster_size=3, max_cluster_size=100.
# None of the pre-existing single-month null runs did -- they used
# min_cluster_size=5 and max_cluster_size=300 (205 for bootstrap/2007-02, and 50
# for bootstrap/2007-05). Audited 2026-10-27; see response.md under R2.8.

UMAP_KW = dict(n_neighbors=15, n_components=10, metric='euclidean',
               init='spectral', min_dist=0.1, spread=1.0)
HDBSCAN_KW = dict(min_cluster_size=3, min_samples=None,
                  cluster_selection_epsilon=0.0, max_cluster_size=100,
                  metric='euclidean')


def align_units(C, seed, align_dim=10, umap_kw=None, hdbscan_kw=None):
    """UMAP + HDBSCAN on a centroid matrix, published settings. -> (labels, Cu).

    Seeding UMAP forces it single-threaded; the published run was unseeded.
    """
    import umap
    from sklearn.cluster import HDBSCAN
    kw = dict(umap_kw or UMAP_KW, n_components=align_dim, random_state=seed)
    Cu = umap.UMAP(**kw).fit_transform(C)
    model = HDBSCAN(**(hdbscan_kw or HDBSCAN_KW)).fit(Cu)
    return model.labels_, Cu


def drift_table(C, labels, k, n_perm, seed, min_size=5):
    """delta/Lambda/p/q per tested group, using WS1's locked call.

    Always returns the full schema, so callers can concatenate even when no
    group reaches min_size. n_perm should be 1e5: at 1e4 the smallest attainable
    p (1e-4) exceeds BH's tightest threshold at ~589 tests (8.5e-5).
    """
    import pandas as pd
    import drift
    rows = []
    for coi in drift.tested_groups(labels, min_size=min_size):
        X, per = drift.group_trajectory(C, labels, coi, k=k, within_month='asis')
        T = X.shape[0]
        r = max(1, min(5, (T - 2) // 2))
        lam, p = drift.perm_test(drift.group_basis(X, r), drift.lam_batch,
                                 n_perm=n_perm, rng=[seed, int(coi)])
        rows.append({'coi': int(coi), 'T': T, 'r': r,
                     'first_period': int(per[0]), 'last_period': int(per[-1]),
                     'delta': float(drift.displacement(X)),
                     'lam': float(lam), 'p_lam': float(p)})
    cols = ['coi', 'T', 'r', 'first_period', 'last_period', 'delta', 'lam',
            'p_lam', 'q_lam']
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df['q_lam'] = drift.bh_qvalues(df.p_lam.values)
    return df[cols]
