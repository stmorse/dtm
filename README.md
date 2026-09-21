# dtm — comment-level topic drift in embedding space

Code for *Comment-level topic drift analysis in the Reddit corpus: an embedding-based approach* (Morse, Runfola, Ford; under revision at PLOS One).

We embed every Reddit comment from 2006–2022 (Pushshift `RC_*` dumps, about 12.7B comments after filtering) with a pretrained sentence transformer. Next we cluster the embeddings within each calendar month and align similar monthly clusters across time into *topic groups*. Finally we measure how those groups move through embedding space, both individually (drift) and relative to each other (inter-topic drift).

## Pipeline

All scripts live in `src/`. They read `../config.ini` (see [Configuration](#configuration)), so **run them from inside `src/`**.

| Stage | Current script | Input → output | Superseded / alternatives |
|---|---|---|---|
| 1. Embed | `1_embed.py` | `RC_{YYYY}-{MM}.{bz2,xz,zst}` → `embed_path/embeddings_{YYYY}-{MM}.zarr` (N × 384, float32) | `1a_convert_embed.py` converts legacy `.npz` embeddings to `.zarr` |
| 2. Cluster (per month) | `2c_cluster.py` | embeddings → centroids, labels, TF-IDF keywords (see layout below) | `2_cluster.py` (npz only), `2a_cluster_s.py` (zarr, streaming), `2b_r_cluster.py` (full KMeans, experimental/broken) |
| 2b. WCSS for choosing k | `2c_wcss.py` | embeddings + saved centroids/labels → `wcss_{YYYY}-{MM}.json` | — |
| 3. Align | `3_align.py` | all monthly centroids → UMAP (10-d) → HDBSCAN topic groups | — |
| 4. Drift test | `7_rw.py` | aligned groups + centroids → per-group displacement, diameter, Λ, permutation p | Main-corpus version currently lives in `notebooks/viz.ipynb` |
| Null / stability | `5a_dist_singlemonth.py` | one month clustered R times → aligned → (then `7_rw.py`) | `5_singlemonth.py`, `5a_r_singlemonth.py` (FAISS Lloyd; broken) |

Details:

- **Embed.** `1_embed.py` streams one monthly dump at a time and skips entries with no `body` or with author `[deleted]`. It truncates each body to 2000 characters and encodes in fp16 across every visible GPU (`start_multi_process_pool`). The results are appended to a zarr array in chunks of `chunk_size` rows. The default model is `sentence-transformers/all-MiniLM-L6-v2` (d = 384). No per-comment metadata is saved. Row *i* of a month's zarr can only be mapped back to text by re-streaming the raw file with the same filter (`utils.read_sentences`).
- **Cluster.** `2c_cluster.py` fits one model per month: `MiniBatchKMeans` via `partial_fit` over chunks (`--model mbkm`), or full `KMeans` (`--model km`, small months only). It then labels every row and computes TF-IDF keywords from the `--top-k` comments nearest each centroid (`--tfidf 1`). The paper uses k = 50.
- **Align.** `3_align.py` stacks all T·k centroids, reduces them with UMAP to `--align_dim` dimensions (plus a separate 2-d UMAP for plotting), and clusters them with HDBSCAN (`min_cluster_size=3`, `max_cluster_size=100`). Each HDBSCAN cluster is a topic group; label `-1` is noise. Row index = `period * k + cluster`.
- **Drift.** `7_rw.py` computes each group's diameter and first-to-last displacement in the full 384-d space. It also computes a log-likelihood-ratio statistic Λ on PCA-projected steps, with an empirical p-value from shuffling the time order of the group's centroids. As written, `7_rw.py` targets the single-month null experiment; the version used for the main corpus is in `notebooks/viz.ipynb`.
- **Null / stability.** `5a_dist_singlemonth.py` clusters a single month `--n_resamples` times, treats the resamples as pseudo time periods, and aligns them. Any drift detected in the result is spurious (paper Appendix S2).

### Output layout

Results are written under `save_path/<subpath>/` (e.g. `mbkm_50/`), with an optional `--special` subdirectory used for k sweeps:

```
models/model_cc_{YYYY}-{MM}.npz     # 'cc': k x d centroids
models/stats_{YYYY}-{MM}.pkl        # inertia, size, num_chunks (2c)
models/wcss_{YYYY}-{MM}.json        # exact WCSS (2c_wcss)
labels/labels_{YYYY}-{MM}.npz       # 'labels': cluster id per embedding row
tfidf/tfidf_{YYYY}-{MM}.pkl         # {'tfidf': {i: {'sample_indices', 'keywords'}}, 'full': {...}}
align/cu.npz, cu2d.npz              # UMAP-reduced centroids (align_dim, 2-d)
align/align_model_HDBSCAN.pkl       # fitted HDBSCAN
align/align_model_HDBSCAN_labels.npz
tab/table_{YYYY}-{MM}.csv           # 7_rw output (single-month case)
```

## Configuration

Copy `config.example.ini` to `config.ini` in the repo root and fill in the paths. `config.ini` is gitignored.

```ini
[general]
data_path  = ...   # directory of raw RC_YYYY-MM.{bz2,xz,zst} files
embed_path = ...   # embeddings_YYYY-MM.zarr
meta_path  = ...   # (unused; metadata is not currently saved)
save_path  = ...   # root for clustering/alignment results
hf_path    = ...   # HF_HOME cache for the sentence-transformer

[embed]
chunk_size = 1000000
hf_model   = sentence-transformers/all-MiniLM-L6-v2
```

Year/month arguments form a Cartesian product: `--start-year 2015 --end-year 2016 --start-month 1 --end-month 3` processes Jan–Mar of *each* year. CLI flag styles differ between scripts. `1_*`/`2*_*` use hyphens (`--start-year`); `3_align`, `5a_*`, and `7_rw` use underscores (`--start_year`, `--sub_path`).

## Running

**Local environment** (Python 3.12, `uv`): `uv pip install -r requirements-local.txt --torch-backend cu128`. This matches the cluster image's pins except for torch, which is built for newer GPUs.

Heavy jobs run as Kubernetes pods using the image `ghcr.io/stmorse/dtm:latest` (`Dockerfile`: PyTorch 2.5.1 / CUDA 12.4 + `requirements.txt`). GitHub Actions rebuilds the image when `Dockerfile`, `requirements.txt`, or the workflow file changes. Code is **not** baked into the image; pods run it from an NFS-mounted checkout. Example manifests are in `manifests/`, e.g. `embed-create.yml` (4 GPUs) and `cluster-create.yml`/`mult-cluster-create.yml` (CPU).

Example (from `src/`):

```bash
python 1_embed.py   --start-year 2022 --end-year 2022 --start-month 10 --end-month 12
python 2c_cluster.py --subpath mbkm_50 --start-year 2006 --end-year 2022 \
                     --start-month 1 --end-month 12 --n-clusters 50
python 3_align.py   --subpath mbkm_50 --start_year 2006 --end_year 2022 --n_clusters 50
```

## Notebooks

- `viz.ipynb`: main figures and tables. It covers topic-group UMAP plots, trajectories, displacement tables, inter-topic distance-change heatmaps, and the permutation LRT (paper Table 1, Figs 2–6).
- `sandbox2.ipynb`: WCSS/elbow plots for choosing k (paper Figs S1–S2).
- `sandbox.ipynb`, `align.ipynb`, `scratch.ipynb`: exploratory work (single-month null, alternative alignments, toy random-walk test).

## Data

The raw data is the Pushshift Reddit comment dumps (Baumgartner et al., ICWSM 2020), which are not redistributed here. Derived artifacts (monthly centroids, keywords, topic-group labels) will be deposited in a public repository with the paper.

## License

MIT (see `LICENSE`).
