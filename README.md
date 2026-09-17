# dtm


## Methodology

This repo contains scripts and notebooks for a research project studying topic drift in embedding space, with an application in the Reddit corpus.  The methodology consists of a pipeline (see `src/`):

1. **Embed.**  Input: raw text, Output: sentence embeddings.  In this case, raw text is Reddit comments 2007-2023, embedding is an off-the-shelf pre-trained sentence embedder applied to the entire comment. 
    - `src/1_embed.py` handles all raw input filetypes.  Default embedder is `all-miniLM-L6-v2`.
    - `1a_convert_embed.py` converts `.npz` to `.zarr` via `dask.array`.

2. **Cluster.** Input: embeddings (time windowed), Output: topic centroids (time windowed).  We use month as window, and mini-batch k-means for clustering.  We also compute the tf-idf keywords for each cluster for interpretive purposes only. This is a 3-part process:
    - First pass: Fit clusters over the chunked data
    - Second pass: Label resulting clusters
    - Partial third pass: Compute tf-idf on representative sentences from each cluster (close to centroid).
    - Save.

    There are several scripts associated with this step.
    - `2_cluster.py` performs these steps for `.npz` embeddings
    - `2a_cluster_s.py` for `.zarr` embeddings.
    - `2b_r_cluster.py` fits deterministic (non-minibatch) KMeans, but only works for smaller years.
    - `2c_cluster.py` combines behavior of above 3 scripts and is preferred script. It also includes a model inertia scoring block

    There is also a `2c_wcss.py` script which computes Within-Cluster Sum-of-Squares for precomputed clusters (involves loading all embeddings and cluster model).  Using `2c_cluster.py` computes this in-flight and is more efficient.

3. **Align.** Input: windowed topic centroids, Output: aligned topic *groups* (i.e. topic centroids over time that are very similar to each other).  This consists of two steps:
    - Dimensionality reduction of topic centroids (using UMAP)
    - "Alignment" via clustering on the reduced centroids (using HDBSCAN)

    The `3_align.py` script handles both.


## Additional scripts

- **Permutation testing.**  The `5_singlemonth.py` script and its counterparts implement a permutation null testing procedure for the above pipeline.  Specifically,
    - Select a single month of embeddings.
    - Apply Step 2 to M instances of the month, with each instance a time-shuffled version of the original ordering.
    - Apply Step 3 to this "sequence" of centroids.
    - Analyze the resulting topic groups for drift (any drift is purely due to noise, since it is the same time window).

    The `5a_dist_singlemonth.py` is a slightly more efficient (?) implementation, but must be Minibatch KMeans and only for `.zarr` embeddings.  The `5a_r_singlemonth.py` uses a deterministic KMeans but is infeasible for larger datasets.

- **Random walk testing.**  The `7_rw.py` script performs a random-walk-based log-likelihood ratio test on aligned topic groups. It consists of:
    - Load topic group labels and full dimension centroids
    - Reduce centroids using PCA
    - For each topic group, 
        - compute "diameter" and "displacement" of the group
        - compute LRT score based on shuffling the time order of the centroids
