# CLAUDE.md

Research repo: topic drift of Reddit comments (2006–2022) in sentence-embedding space. The pipeline (embed → monthly k-means → UMAP+HDBSCAN alignment → drift tests), the scripts and the output layout are described in `README.md`.

**Current focus: PLOS One major revision, due 2026-11-13.** The shared working document is **`refs/revision/response.md`**. It holds the workstream table, verified facts, per-comment plans and status, response text, manuscript edits, and links to new figures and tables. Read it at the start of every session.

## Ground rules

- **No git write actions.** Don't commit, push, pull, branch, stash, or checkout; the user does all git operations. Read-only git (`status`, `diff`, `log`, `show`) is fine.
- **Package manager: `uv`.** Use `uv pip install ...` into `.venv/`, and `uv run` / `.venv/bin/python` to execute. Don't use bare `pip` or conda.
- **No cluster launches.** kubectl isn't available on prospero. Write or modify manifests in `manifests/`; the user launches them on cm.
- The manuscript source is on Overleaf and is **not** in this repo. Never edit the PDF. Put proposed text in `response.md` under **Manuscript changes**; the user applies it by hand.
- Don't modify the published results under `save_path/mbkm_50/` or `save_path/km/`. Write new results elsewhere (see Outputs).
- The corpus contains offensive and extremist text. Don't reproduce raw comments in outputs or docs unless it's necessary.

## Parallel sessions (workstreams)

The revision is split into workstreams WS1–WS5, defined in `response.md`. Several sessions may run at once in this same working tree, so:
- Edit only your workstream's comment sections in `response.md`, plus your row in the workstream table. Findings that affect others go under *Cross-workstream notes* (dated).
- New scripts are numbered per workstream: WS1 → `src/8*_*.py`, WS2 → `src/9*_*.py`, WS3 → `src/10*_*.py`. New notebooks are `notebooks/rev_ws<N>_*.ipynb`. Don't edit another workstream's files.
- `src/drift.py` (owned by WS1) holds the shared drift statistic, and other workstreams import it. Don't change pipeline scripts `0_*`–`7_*` or `utils*.py`. If a fix is needed there, note it in *Cross-workstream notes*.
- Every new script takes `--seed` (default 0), writes the seed and parameters alongside its outputs, and reads paths from `config.ini` (not hardcoded).

## Outputs

| What | Where |
|---|---|
| Final figures for the manuscript/response | `refs/revision/figs/<commentID>_<name>.{png,pdf}` (tracked) |
| Final tables | `refs/revision/tables/<commentID>_<name>.{csv,md}` (tracked) |
| Local intermediates | `output/ws<N>/` (gitignored) |
| Large results and cm job outputs | `save_path/rev/ws<N>/` on sciclone |

## Machines and environment

| Machine | Role |
|---|---|
| **prospero** (this machine) | Local editing, notebooks, and light-to-medium compute. RTX 5090 (Blackwell, sm_120), Python 3.12, `.venv/`. |
| **cm** | Kubernetes cluster for heavy jobs (full-month embedding and clustering). GPUs are crowded. |
| **sciclone** | W&M HPC storage; all data and results live here. Mounted on prospero at `/mnt/sciclone` via **sshfs**. |
| **bora** | Old sciclone front-end; unreliable, not used. |

- **Local env:** `uv pip install -r requirements-local.txt --torch-backend cu128`. `requirements.txt` is the cm image's pin set (torch 2.5.1/cu124, no Blackwell support); `requirements-local.txt` matches it except for torch.
- **The sshfs mount is slow for bulk reads.** Centroids, labels, tfidf, and align outputs (hundreds of MB) are fine. Don't stream full months of embeddings locally (late months hold ~200M × 384 float32, about 300 GB uncompressed); build subsamples on cm instead.
- **The cm loop:** user pushes → `git pull` in the sciclone checkout → `kubectl apply -f manifests/<job>.yml`.
  - Pods run code from the NFS checkout, not from the image. Checkouts: `/sciclone/home/stmorse/projects/dtm` (newer manifests) and `/sciclone/geograd/stmorse/dtm` (older ones). Check which one a manifest `cd`s into.
  - Inside pods, paths have no `/mnt` prefix, and pods need a `config.ini` in their checkout with pod paths (`config.example.ini`).
  - The image `ghcr.io/stmorse/dtm:latest` rebuilds only when `Dockerfile`, `requirements.txt`, or the workflow file changes.

## Paths

`save_path` = `/mnt/sciclone/proj-ds/geograd/stmorse/reddit` (locally; `/sciclone/...` on cm).

- Raw dumps: `/mnt/sciclone/data10/twford/reddit/reddit/comments/RC_YYYY-MM.{bz2,xz,zst}`
- Embeddings: `save_path/embeddings/embeddings_YYYY-MM.zarr` (some `.npz`)
- Monthly comment counts: `save_path/totals.csv`
- Published run: `save_path/mbkm_50/{models,labels,tfidf}` and `mbkm_50/align/max_100/`
- k grid: `save_path/km/km{30,50,70,90}`; elbow sweep `km/kmm/{k}`; single-month nulls `km/kmp*`, `mbkm_50/bootstrap`
- Older root still referenced by some code: `/sciclone/geograd/stmorse/...`

## Conventions and gotchas

- Scripts read `../config.ini` (gitignored; template is `config.example.ini`). **Run them from `src/`.**
- CLI flag style is inconsistent. `1_*`/`2*_*` use hyphens; `3_align`, `5a_*`, and `7_rw` use underscores (`--sub_path` in 7_rw vs `--subpath` elsewhere). Year × month ranges form a Cartesian product.
- Centroid matrix convention: row = `period * k + cluster`; HDBSCAN label `-1` = noise. Group IDs in the notebooks are valid only for `align/max_100`.
- Known issues (details in `response.md` → Cross-cutting issues):
  - `2c_cluster.py` / `5a_dist` drop the trailing partial batch (does not affect the paper).
  - For mbkm, `inertia_` covers only the last batch.
  - Nothing is seeded.
  - `np.amax(labels)` is used as a group count (off by one).
  - `7_rw.py` / viz.ipynb shuffle centroids rather than steps (see R1.1).
  - `5a_dist` resamples vary only the init.
  - `2b_r_cluster.py`, `5a_r_singlemonth.py`, and `test-create.yml` are broken.
- `1_embed.py` keeps `[removed]`/`[deleted]` bodies from live authors, truncates at 2000 characters, and saves no metadata. Embedding row *i* maps to text only by re-streaming the raw file with the same filter (`utils.read_sentences`).
