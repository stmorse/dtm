# Response to Reviewers — shared working document

**PLOS One major revision, due 2026-11-13.** *Comment-level topic drift analysis in the Reddit corpus: an embedding-based approach.*

This file is the single landing spot for the revision. It holds workstream status, plans, response-letter text, manuscript edits, and links to new figures and tables. The user applies manuscript edits by hand (the source is on Overleaf, not in this repo).

**How to use it (all sessions):**
- Each comment section has an **[internal]** block (workstream, status, plan, notes) and three letter blocks (**Response**, **Manuscript changes**, **Figures / tables**). Delete the [internal] blocks when producing the final letter.
- Edit only your workstream's comment sections and your row in the workstream table. Findings that affect another workstream go under *Cross-workstream notes* at the bottom.
- **Status values:** `todo` → `designing` (plan proposed, awaiting user) → `running` → `drafted` (text ready for user edit) → `done`.
- **Manuscript changes format:** `§/Eq/Fig — Old: "…" → New: "…"` (LaTeX OK). Line numbers refer to the v2 PDF (`refs/Topic_drift_manuscript_v2.pdf`).
- New figures go in `refs/revision/figs/`, new tables in `refs/revision/tables/` (CSV + markdown). Name them with the comment ID, e.g. `R1.1_table1_revised.csv`.

---

## Workstreams

| WS | Scope | Deliverables | Depends on | Status |
|---|---|---|---|---|
| **WS1** Drift test | R1.1, R2.5, R2.6, Fig 5 test for R2.9 | `src/drift.py` (reusable statistic), `src/8_*.py`; revised Table 1, Fig 4, (Fig 5); rewritten §2.6 and S3 | — (**start first; others consume `src/drift.py`**) | todo |
| **WS2** Alignment & k sensitivity | R2.4, R2.2, R1.3 | `src/9_*.py`; SI section and figures | WS1 statistic (can start on partition stability without it) | todo |
| **WS3** Scale & model robustness | R2.3, R2.1, R2.8, R1.2 (normalization) | `src/10_*.py`, `manifests/rev-*.yml`; subsample dataset on sciclone; SI sections | WS1 statistic; cm time (start the cm jobs early) | todo |
| **WS4** Writing | R1.4, R2.9, R1.2 (citations), R2.7, E4, E6, R1.6 (text), cross-cutting text fixes | Manuscript changes and responses only (no code) | WS1 for final numbers in the text | todo |
| **WS5** Release & admin | E1, E2, E3, E5, R1.5, R2.10 (tabled) | Zenodo deposit, code release, hyperparameter table | Everything else (do last) | todo |

## Verified shared facts (as of 2026-09-21)

- **Published results:**
  - Clustering: `save_path/mbkm_50/` (models, labels, tfidf; 204 months, Jan–Feb 2025).
  - Alignment: `mbkm_50/align/max_100/` (HDBSCAN `max_cluster_size=100`; Feb 3 2025). `align/max_20/` is an alternative that wasn't used.
  - Table 1 and Fig 4 were computed in `notebooks/viz.ipynb` (cell 75: PCA r=45, B=10,000, unseeded).
- **Clustering provenance:** `mbkm_50` predates `2c_cluster.py` (created 2025-09-04). 2007–2014 came from `2_cluster.py` (`model_*.pkl` present); 2006 and 2015–2022 from `2a_cluster_s.py`. Both fit every chunk, so **the 2c dropped-batch bug does not affect the paper**.
- **k grid:**
  - `km/km{30,50,70}`: 204 months of centroids each; `km90`: 192 months.
  - Methods: KM for 2006–2013, MBKM for 2016–2022, mixed for 2014–15 (per `km/readme.txt`; this is why Fig S1 excludes 2014–15).
  - `wcss_*.json` exists for 72 months only.
  - `km/kmm/{k}` is the 2017-03 elbow sweep. `km/kmp*` are single-month null runs.
- **Monthly volumes:** `save_path/totals.csv` (2006-01 has 2,773 comments).
- **HDBSCAN group IDs** hardcoded in the notebooks (e.g. 44, 811, 712) are valid only for `align/max_100`.

## Cross-cutting issues (found internally, not raised by reviewers)

These must be fixed for accuracy. The owner is WS4 unless stated.

- [ ] Truncation: the MS says "at most 128 tokens". The code truncates bodies at 2000 characters; the model's max_seq_length is 256 tokens. (WS3 to confirm the model config.)
- [ ] `[removed]` and `[deleted]` *bodies* by live authors are kept. Check whether they form artifact clusters (WS3), then describe them accurately in §2.1.
- [ ] S1: WCSS is described as "average distance" but the formula uses squared distances. The 2014–15 exclusion needs an explanation (mixed KM/MBKM, see above).
- [ ] Limitations cites "Appendix 4.2"; it should be S2.
- [ ] S2 says "resampled with replacement"; the code permutes and varies the init only (WS3 fixes this under R2.8).
- [ ] δ_k depends on two noisy endpoints and grows with group lifespan. Consider a trend-based displacement (WS1).
- [ ] Code bugs, **not** affecting the paper: `2c` dropped batch, mbkm `inertia_`, `np.amax(labels)` group count (off by one; appears in `3_align.py`'s results.log and some notebook plots; the Table 1 cell iterates `np.unique`, so the table is unaffected).

---

## Journal requirements (Editor)

### E1

> Please ensure that your manuscript meets PLOS ONE's style requirements, including those for file naming. The PLOS ONE style templates can be found at https://journals.plos.org/plosone/s/file?id=wjVg/PLOSOne_formatting_sample_main_body.pdf and https://journals.plos.org/plosone/s/file?id=ba62/PLOSOne_formatting_sample_title_authors_affiliations.pdf

**[internal]** Workstream: WS5 · Status: todo  
Plan: Check the manuscript against the PLOS templates (title page, headings, file names) at submission time.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### E2

> Please note that PLOS One has specific guidelines on code sharing for submissions in which author-generated code underpins the findings in the manuscript. In these cases, we expect all author-generated code to be made available without restrictions upon publication of the work. Please review our guidelines at https://journals.plos.org/plosone/s/materials-and-software-sharing#loc-sharing-code and ensure that your code is shared in a way that follows best practice and facilitates reproducibility and reuse.

**[internal]** Workstream: WS5 · Status: todo  
Plan: Clean repo, tagged GitHub release, Zenodo DOI through the GitHub integration; cite it in the Code Availability statement.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### E3

> Thank you for uploading your study's underlying data set. Unfortunately, the repository you have noted in your Data Availability statement does not qualify as an acceptable data repository according to PLOS's standards. At this time, please upload the minimal data set necessary to replicate your study's findings to a stable, public repository (such as figshare or Dryad) and provide us with the relevant URLs, DOIs, or accession numbers that may be used to access these data. For a list of recommended repositories and additional information on PLOS standards for data deposition, please see https://journals.plos.org/plosone/s/recommended-repositories.

**[internal]** Workstream: WS5 · Status: todo  
Plan: Deposit derived data on Zenodo or figshare: monthly centroids for mbkm_50 (204×50×384, ~15 MB), TF-IDF keywords, topic-group labels, per-group statistics, WCSS grid. Raw Pushshift data is cited, not redistributed.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### E4

> Please ensure that you refer to Figure 1 in your text as, if accepted, production will need this reference to link the reader to the figure.

**[internal]** Workstream: WS4 · Status: todo  
Plan: Fig 1 is never cited in the text. Add "(Fig 1)" in §2.2 (methodological framework).

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### E5

> We notice that your supplementary figures are uploaded with the file type 'Figure'. Please amend the file type to 'Supporting Information'. Please ensure that each Supporting Information file has a legend listed in the manuscript after the references list.​

**[internal]** Workstream: WS5 · Status: todo  
Plan: Submission-system fix: change the SI file type to "Supporting Information"; add SI legends after the references.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### E6

> If the reviewer comments include a recommendation to cite specific previously published works, please review and evaluate these publications to determine whether they are relevant and should be cited. There is no requirement to cite these works unless the editor has indicated otherwise.

**[internal]** Workstream: WS4 · Status: todo  
Plan: Covers R1.2 (second half) and R2.7. Decline the suggested citations as outside scope; broaden related work with relevant semantic-change literature instead.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

## Reviewer 1

### R1.1

> The permutation test in Appendix S3 needs one more look. If the authors only shuffle the order of the step vectors \Delta_t, their mean and covariance do not change, so the reported \Lambda should also stay unchanged. This does not quite add up with the reported p-values; please clarify the actual permutation procedure and correct the text or analysis if needed.

**[internal]** Workstream: WS1 · Status: todo  
Plan: The appendix text is wrong: the code (viz.ipynb cell 75, src/7_rw.py) shuffles the *centroids*, not the steps Δ_t. Deeper problems follow: the mean step telescopes to (c_last−c_first)/(T−1), so Λ is an endpoint Mahalanobis distance; the null is exchangeability (static topic + noise), not a random walk, so a driftless RW would also be rejected; Σ̂ in r=45 dims from ~9 steps is rank-deficient (pinv); same-month duplicates and gaps; p can be 0; no seed. **Plan:** (1) reproduce the published Table 1 / Fig 4 numbers exactly as a baseline; (2) reframe as a time-label permutation test for temporal trend using every member and real month indices (e.g. the norm of the OLS slope of c(t) on t, possibly with a shrinkage covariance and r < T); p = (1+b)/(B+1); seeded; (3) optionally a separate drift-vs-RW check (variance ratio); (4) recompute Table 1 and Fig 4; rewrite §2.6 and S3. Put the reusable statistic in `src/drift.py` for WS2/WS3.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R1.2

> Please clarify whether the MiniLM embeddings are L2-normalized before k-means, centroid-distance calculation, and drift analysis. Euclidean distance on unnormalized sentence embeddings can be affected by vector norm, so this small detail matters here. The related work on semantic representation could be slightly broadened. Studies such as ROUGE-SEM: Better evaluation of summarization using ROUGE combined with semantics, From coarse to fine: Enhancing multi-document summarization with multi-granularity relationship-based extractor, and Towards Curriculum Learning of Multi-Document Summarization Using Difficulty-Aware Mixture-of-Experts may provide some broader perspectives on semantic-aware evaluation and representation learning. These works are not direct topic-drift baselines, of course, but may be useful as methodological background.

**[internal]** Workstream: WS3 (normalization) / WS4 (citations) · Status: todo  
Plan: Normalization: all-MiniLM-L6-v2 ends in a Normalize module, so embeddings should be unit-norm (verify on a stored zarr chunk: norms ≈ 1 up to fp16). Centroids are *not* unit-norm; check that the main results hold with re-normalized centroids / cosine distance (cheap, uses the WS1 statistic). Citations: see E6.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R1.3

> The fixed choice of k=50 for all 204 monthly windows is reasonable for scalability, but the corpus size changes dramatically over time. A short sensitivity discussion would help, especially for the very early low-volume years.

**[internal]** Workstream: WS2 · Status: todo  
Plan: Handle jointly with R2.2. Early years: show WCSS per k for 2006–2009 specifically, and drift for early-born groups across k.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R1.4

> Some conclusions are a bit too strong. Phrases such as “reflects durable shifts in meaning” or “evidence for polarization” go beyond what embedding-space movement alone can prove. I would tone these claims down a little.

**[internal]** Workstream: WS4 · Status: todo  
Plan: Handle jointly with R2.9.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R1.5

> The reproducibility part could be cleaner. Please provide the analysis code, main hyperparameters/seeds, and preferably the derived monthly centroids/topic trajectories, since reproducing the full 12.7-billion-comment pipeline is obviously not a piece of cake.

**[internal]** Workstream: WS5 · Status: todo  
Plan: Handle jointly with E2 and E3. Add an SI table of all hyperparameters and seeds (k, chunk size, MBKM settings, UMAP/HDBSCAN params, PCA r, B).

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R1.6

> A few figures are rather hard to read, especially the dense UMAP labels and heatmaps. Please enlarge the labels/legends and make clear again that the 2-D UMAP plots are only visualization, while the reported drift statistics are calculated in the original embedding space.

**[internal]** Workstream: WS4 (text) / WS1 (regen) · Status: todo  
Plan: Regenerate Figs 2–6 with larger labels and legends (the figures that depend on the drift test are regenerated by WS1). Add a caption sentence: "UMAP projection for visualization only; all statistics are computed in the original 384-d space."

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

## Reviewer 2

### R2.1

> The whole analysis relies on all-MiniLM-L6-v2. It would be useful to repeat part of the experiment with another sentence-embedding model to show that the observed drift is not model-specific.

**[internal]** Workstream: WS3 · Status: todo  
Plan: Full re-embedding of 12.7B comments is infeasible. Plan: sample ~1k comments per (month, cluster), recover their text (embedding row i ↔ `utils.read_sentences` filter order), and re-embed with a second model (e.g. all-mpnet-base-v2 or gte-base) on the 5090. Keep the MiniLM cluster/group assignments, recompute each group's trajectory under model B, and report rank correlation of drift statistics and agreement of the significant set. Optionally rerun the pipeline end-to-end on the subsample under model B.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.2

> A fixed k = 50 is used for every month, although the monthly corpus size changes dramatically over time. The authors should provide a stronger sensitivity analysis for different values of k.

**[internal]** Workstream: WS2 · Status: todo  
Plan: The existing k grid covers all months: km/km{30,50,70} have 204 months of centroids, km90 has 192. Align each (3_align settings, seeded), match topic groups across k (centroid overlap by month), and compare the drift ranking and significant set (WS1 statistic). Also note the km/km50 run is an independent k=50 reclustering (KM for 2006–13, MBKM for 2016–22), a free replicate of mbkm_50.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.3

> The corpus size in recent years is more than 160 times that of the early period. This imbalance may affect centroid stability and measured drift. A controlled subsampling analysis would be helpful.

**[internal]** Workstream: WS3 · Status: todo  
Plan: Build a seeded, fixed-N-per-month subsample of embeddings (N ≈ smallest usable month; 2006-01 has only 2,773 comments, see `totals.csv`, so perhaps from 2007 on or with N ~ 50–100k plus a separate early-years caveat). Save the row indices. Recluster (k=50), realign, recompute drift; compare with the full-data results. This subsample also serves R2.1, R2.2, R2.8.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.4

> Topic alignment depends on UMAP followed by HDBSCAN. Since both steps may influence the resulting trajectories, sensitivity to UMAP/HDBSCAN parameters and random seeds should be reported.

**[internal]** Workstream: WS2 · Status: todo  
Plan: Cheap: 10,200 × 384 centroids. Grid over UMAP n_neighbors {10,15,30,50}, min_dist {0,0.1,0.25}, align_dim {5,10,20}, HDBSCAN min_cluster_size {3,5,10}, × 10 seeds. Report ARI/AMI against the published partition, the number of groups, the noise fraction, and stability of per-group drift for the Table 1 topics. Add a no-UMAP baseline (e.g. agglomerative or HDBSCAN on cosine distance between centroids).

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.5

> The random-walk test assumes that topic-step vectors are i.i.d. Gaussian. This is a strong assumption for temporally evolving discourse and deserves further justification or diagnostic testing.

**[internal]** Workstream: WS1 · Status: todo  
Plan: The permutation test does not assume Gaussianity; the Gaussian form only motivates the statistic. Report diagnostics: lag-1 autocorrelation of steps (the MA(1) signature of independent clustering noise ≈ −0.5), normality of PCA-projected steps. Frame the test as an exchangeability null. Joint with R1.1.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.6

> The analysis produces more than 1,000 aligned topic groups and reports significance using p<0.05. A multiple-testing correction such as FDR should be considered before identifying significantly drifting topics.

**[internal]** Workstream: WS1 · Status: todo  
Plan: Apply Benjamini–Hochberg FDR across all tested groups; report counts at q < 0.05 and q < 0.01; use q-values in Table 1 and Fig 4.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.7

> The related work on temporal semantic representation could be broadened a little. Some studies such as An efficient loss function and deep learning approach for ranking stock returns in the absence of prior knowledge, A hierarchical deep model integrating economic facts for stock movement prediction, and Separating the predictable part of returns with CNN-GRU-attention from inputs to predict stock returns, may provide some broader methodological perspectives.

**[internal]** Workstream: WS4 · Status: todo  
Plan: Decline politely (stock-return prediction is outside scope); point to the broadened related work (E6).

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.8

> The current clustering robustness analysis is mainly based on one selected time window. Testing several early, middle, and late periods would provide stronger evidence of stability.

**[internal]** Workstream: WS3 · Status: todo  
Plan: Current resamples (`5a_dist`) differ only by k-means++ init; minibatch updates are order-invariant within a batch. Implement true bootstrap or subsample resampling; run the stability null on early / middle / late months (e.g. 2009-06, 2015-06, 2021-06) on cm. Report the false-positive rate of the WS1 test under the null.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.9

> Some interpretations are stronger than the analysis supports. Changes in embedding geometry indicate changes in Reddit discourse representations, but statements about cultural realignment, polarization, or durable shifts in meaning should be phrased more cautiously.

**[internal]** Workstream: WS4 (+WS1 for Fig 5 test) · Status: todo  
Plan: Soften the abstract, §3.3, §4, and §4.2 (no "durable shifts in meaning", "polarization", "cultural realignment"; say "changes in the embedding geometry of Reddit discourse"). Also: the abstract claims inter-topic change "beyond what the null model can explain", but Fig 5 has no null or uncertainty. Either WS1 adds a test (slope of pairwise distance vs time with permutation or bootstrap CI) or the claim is removed.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.10

> The Ethics Statement is currently listed as N/A. Given that the study analyzes billions of user-generated Reddit comments, a short discussion of privacy, public-data use, deleted users, and responsible handling of sensitive content would be appropriate.

**[internal]** Workstream: WS5 — **TABLED** · Status: tabled  
Plan: Tabled by the user (IRB determination pending). Draft points for later: Pushshift public data; deleted authors removed; no usernames, IDs or metadata retained (embeddings only); offensive content analysed but not reproduced.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

---

## Cross-workstream notes

_Append dated notes here when a finding in one workstream affects another._
