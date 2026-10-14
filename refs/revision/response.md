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
| **WS1** Drift test | R1.1, R2.5, R2.6, Fig 5 test for R2.9 | `src/drift.py` (reusable statistic), `src/8_*.py`; revised Table 1, Fig 4, (Fig 5); rewritten §2.6 and S3 | — (**start first; others consume `src/drift.py`**) | running: steps 1–4, 6, 7 done (2026-09-23); **only 22 groups survive BH-FDR and none of the named topics do — WS4 see R2.6**; next step 8 (Table 1/Fig 4) |
| **WS2** Alignment & k sensitivity | R2.4, R2.2, R1.3 | `src/9_*.py`; SI section and figures | WS1 statistic (can start on partition stability without it) | todo |
| **WS3** Scale & model robustness | R2.3, R2.1, R2.8, R1.2 (normalization) | `src/10_*.py`, `manifests/rev-*.yml`; subsample dataset on sciclone; SI sections | WS1 statistic; cm time (start the cm jobs early) | running: step 1 (quick checks) done 2026-09-22; step 2 (R2.1 retrieval) next |
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
- **Per-comment metadata (index ↔ comment ID) exists for 2006-01 … 2015-10 only** (checked 2026-09-22):
  - `save_path/metadata/metadata_YYYY-MM.csv`: 2006-01 … 2012-12 (84 months, Apr–May 2025). The files are **gzip-compressed despite the `.csv` name**. Columns: `id, created_utc, parent_id, subreddit, subreddit_id, author, idx`. The script that wrote them is not in the repo.
  - `save_path/metadata2/metadata_YYYY-MM.pkl`: 2007-01 … 2015-10 (106 months, Nov 2024, 68 GB). Each file is a list of `[row_idx, author, id, created_utc]`, written by the now-commented-out NPZ path in `1_embed.py`.
  - Row counts match the embeddings and `totals.csv` (checked for 2006-01 and 2007-01).
  - **None for 2015-11 … 2022-12** (86 months, the largest). There, row *i* can be mapped to text only by re-streaming the raw dump with the `1_embed.py` filter.
  - Comment text is in neither file. Retrieving bodies always needs one pass over that month's raw dump. Total dump size is ~1.4 TB compressed; a late month is ~32 GB of `.zst`.
- **HDBSCAN group IDs** hardcoded in the notebooks (e.g. 44, 811, 712) are valid only for `align/max_100`.

## Cross-cutting issues (found internally, not raised by reviewers)

These must be fixed for accuracy. The owner is WS4 unless stated.

- [ ] Truncation: the MS says "at most 128 tokens". The code truncates bodies at 2000 characters; the model's max_seq_length is 256 tokens. **Confirmed by WS3 (2026-09-22):** `max_seq_length` = 256 (`sentence_bert_config.json`, and the loaded model), tokenizer `model_max_length` 256. The 128 figure is probably from the model card's *training* setup ("sequence length limited to 128 tokens"), not inference. So the effective input is the first min(2000 chars, 256 word pieces). In 2008-06 (all 387,632 kept comments; tokens counted on a 50k sample), 0.5% of bodies exceed 2000 chars, 2.1% exceed 256 tokens after the char cut, and 8.0% would exceed 128; the median is 29 tokens. Suggested §2.1 text: "comments were truncated to 2,000 characters and the encoder's 256-token limit". Embedding ran in fp16 (`1_embed.py`); worth a clause too.
- [ ] `[removed]` and `[deleted]` *bodies* by live authors are kept. Check whether they form artifact clusters (WS3), then describe them accurately in §2.1. **WS3 check (2026-09-22, `src/10a_quickchecks.py` → `output/ws3/10a/`): no artifact clusters from these bodies.**
  - Identical bodies embed to one point. Rows equal to the `[removed]`/`[deleted]` vector make up ≤ 1.3×10⁻⁵ of rows (first 400k rows of 2006-06, 2008-06, 2012-06, 2018-06; in 2008-06's full stream there are 0 `[removed]` and 2 `[deleted]` bodies). Pushshift captured most comments before removal.
  - No centroid is near either point: min distance 0.84 across 10,200 centroids, versus a median of 1.02.
  - The clusters whose TF-IDF keywords contain "removed" (139 of 10,200 with score > 0.1, mostly 2013–2022; ≤ 3.8% of a month's comments) are **AutoModerator/bot removal notices** ("…has been removed… message the moderators"), i.e. live bot text, not removed bodies. They form aligned groups (e.g. 561, 575, 599, 582, 568, 609), which is consistent with WS1's note that 561 "compose/message" is a bot template.
  - A second artifact class: **spam/keyboard-mash clusters** with near-duplicate members (centroid norm > 0.6, vs a median of 0.40). There are 226 such clusters, 182 of them in 2019–2022, e.g. groups 910, 8, 899, 7, 14.
  - Suggested §2.1/§4.2 text: deleted-author comments were excluded; bodies removed after collection are rare in the dumps (< 0.01%) and form no cluster; bot and templated text is retained and forms a few recognisable topic groups, which we leave in.
- [ ] S1: WCSS is described as "average distance" but the formula uses squared distances. The 2014–15 exclusion needs an explanation (mixed KM/MBKM, see above).
- [ ] Limitations cites "Appendix 4.2"; it should be S2.
- [ ] S2 says "resampled with replacement"; the code permutes and varies the init only (WS3 fixes this under R2.8).
- [ ] δ_k depends on two noisy endpoints and grows with group lifespan. Its δ² permutation p-value (WS1, R1.1 step 4) addresses noise. Mention the lifespan dependence when interpreting Table 1.
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

**[internal]** Workstream: WS1 · Status: running (step 1 done 2026-09-22)

**Decision (user, 2026-09-22): keep Λ, fix the covariance, correct the framing, and report δ² for comparison.** Don't introduce a new test.

*What is implemented now* (viz.ipynb cell 75 produced Table 1 and Fig 4; `src/7_rw.py` is the single-month version):
- Group members are ordered by row index (period·k + cluster). Steps are consecutive differences.
- PCA r=45 is fit on all 10,200 centroids. Λ = (T−1)/2 · μ̄ᵀ pinv(Σ̂) μ̄, with Σ̂ from the observed-order steps.
- The null shuffles the **centroid** order (not the steps, contrary to Appendix S3), recomputes μ̄, and **reuses the observed pinv(Σ̂)**. Then p = mean(null > obs), with B=10,000, unseeded.

*What it actually tests.* The mean step telescopes, μ̄ = D/(T−1) with D = c_last − c_first. So Λ = Dᵀ Σ̂⁻¹ D / (2(T−1)) is a Mahalanobis length of the endpoint displacement, compared against random pairs of the same group's centroids.
- **Null:** the group's centroids are *exchangeable in time* (a fixed location plus clustering/sampling noise). This is not a random walk. A driftless RW is also rejected, because E‖c_j−c_i‖² ∝ |j−i| and the endpoints are the most time-separated pair.
- **Rejection means:** the topic's position depends on time (drift or accumulated diffusion). It does not mean "directional drift beyond a RW".
- This null matches the S2 single-month resampling experiment.

*Toy simulations* (isotropic Gaussian, 300 reps, p<0.05; `output/ws1/rw_sim*.py`, gitignored, run from that directory):

| Process | Published, T=10 | Own-cov per shuffle, T=10 | Published, T=30 | Own-cov per shuffle, T=30 |
|---|---|---|---|---|
| Static noise (null) | 0.22 | 0.03 | 0.09 | 0.05 |
| Driftless RW | 0.68 | 0.74 | 0.62 | 1.00 |
| RW + drift | 0.85 | 0.88 | 0.92 | 1.00 |

- At r=45, T=10 the published Λ rejects ≈ 0 for every process. Own-cov Λ reaches ≈ 17%. δ² under the same shuffle reaches ≈ 90% (it also detects time-dependence, not drift specifically).
- A sign-flip RW null has ~0 power under clustering noise (steps are MA(1)). So separating drift from diffusion is out of scope; state this as a limitation.

*Implementation plan* (reusable functions in `src/drift.py`, driver `src/8_drift_test.py`, seeded; outputs to `output/ws1/`, final products to `refs/revision/{tables,figs}/R1.1_*`):
1. **Baseline.** Reproduce the published Table 1 and Fig 4 p-values from `mbkm_50` + `align/max_100` using the published procedure. Use a seed and B=10,000; values should match up to Monte Carlo error. Record how many groups were tested (T ≥ 5, label ≠ −1).
2. **Ordering.** Order members by month (period = row // 50). Decide and document how to handle groups with more than one centroid in a month: e.g. average within a month, or keep all with ties broken at random. Report how many groups are affected. T should count months if averaging.
3. **Corrected Λ.**
   - Recompute Σ̂ (and its inverse) inside every shuffle.
   - Reduce dimension so Σ̂ is estimable. Primary choice: global PCA with a fixed small r (e.g. r = 3 so that only groups with T−1 > r are tested), or per-group r_k = min(r_max, T_k − 2). Pick one after the sensitivity check below. Ledoit–Wolf shrinkage at a moderate r is a secondary check.
   - Use p = (1+b)/(B+1).
4. **δ² comparison.** Run the same shuffle on δ² = ‖c_last − c_first‖² in the full 384-d space. Report it as a comparison column.
5. **Sensitivity.** Vary r (e.g. 2, 3, 5, 10, 20) and report the number of significant groups, plus agreement between Λ and δ² (overlap of significant sets, rank correlation of p-values).
6. **Calibration check on real data.** Apply the corrected test to the existing single-month null runs (`km/kmp*`, `mbkm_50/bootstrap`); the rejection rate should be ≈ α. WS3 extends this to other windows under R2.8.
7. **Multiple testing.** BH-FDR (R2.6) and step diagnostics (R2.5) on the final procedure.
8. **Outputs.** Revised Table 1 (δ_k, Λ_k, p_k, q_k, δ²-p_k) and revised Fig 4 (significant vs not, from q-values). Draft the text in "Manuscript changes".

*Step 1 results (2026-09-22).* `src/8a_baseline.py --seed 0` → `output/ws1/8a_baseline/` (`groups.csv`, `compare_published.csv`, `params.json`). Uses `drift.perm_test_published` / `drift.perm_exact_published`.
- **Reproduced.** 1006 non-noise groups; **589 tested** (T ≥ 5). δ and Λ match all 10 rows of Table 1 to ±0.001, with one exception ("conspiracy" Λ, below). Fig 4b "laughing" (coi 774: Λ 3.743, p 0.154) and Fig 4c "religion" (coi 811: Λ 8.295) also match. Λ is identical under the full SVD solver and 5 randomized-PCA seeds, so the unseeded PCA doesn't matter.
- **The null is exactly enumerable.** Because of telescoping, the shuffled Λ depends only on which ordered pair (i, j) ends up at the ends. The permutation distribution is therefore uniform over T(T−1) values, and the exact p is available (no Monte Carlo). MC with B=10⁴ agrees with it (max |Δp| = 0.012).
- **Ties.** The observed pair and its reverse always tie Λ_obs. The published `null > obs` drops them. Whether they drop depends on float rounding, so tie handling is now explicit (relative tolerance).
- **Group ids for Table 1** (align/max_100): conspiracy 290, oil 771, scientific 798, government 949, republican 999; woman 510, music 244, team 146, games 766, delicious 425. Fig 4: laughing 774, religion 811.
- **Discrepancies to resolve:**
  1. "conspiracy" (290): δ 0.387 and p 0.286 match; the printed Λ is 0.277, reproduced 0.505. Probably a transcription error (Λ≈0.277 belongs to coi 351/420, which aren't in the table).
  2. The printed p for scientific (0.003), government (0.004) and religion (0.001) can't be produced by the published convention. The exact P(null > obs) = 0 for all three, and the notebook's own saved output for 811 is 0.0. The inclusive exact p is 0.0074 / 0.0110 / 0.0023. All are < 0.05 either way, so the conclusions are unchanged, but the printed numbers came from some other run.
  3. **"Top and bottom-ranked" is curated.** The top rows are δ-ranks 2, 3, 7, 8, 9: skipped are 53 "numbers/math" (rank 1), 561 "compose/message" (a bot template), 428 "wear", 712 "spelling". The bottom rows are ranks 3, 4, 7, 14, 20. The caption should say "selected" (→ WS4), or the revised table should give strict ranks.
  4. **Fig 4a colours groups by Λ ≥ 5 and T ≥ 10** (viz.ipynb cell 82), not by p < 0.05. That gives 41 groups, of which 36 have p < 0.05. The caption says "significant". The revised Fig 4 will use q-values (step 8).
- **Counts under the published test:** p < 0.05 for 113/589 (exact, strict `>`), p < 0.01 for 80, **p = 0 for 76**. With the conventional inclusive p = P(null ≥ obs), it's 92 at p < 0.05 and 30 at p < 0.01. The inclusive p has a floor of 2/(T(T−1)): 0.10 at T=5, 0.067 at T=6. So groups with T ≤ 6 (213 of 589) can never reach p < 0.05; the strict convention let 8 of them through. Use (1+b)/(B+1), or the exact inclusive p, from step 3 on.
- **Inputs for steps 2–3:** 42 of 589 groups have more than one centroid in some month (T > n_months). 583 of 589 have T − 2 < 45, so Σ̂ is rank-deficient at r = 45 for nearly every group. T distribution: 130 groups at T=5, 83 at T=6, median 8, max 83.

*Step 2 results (2026-09-22): within-month duplicates.* `src/8b_drift_test.py --within_month {asis,random,average}`, global PCA r=3, B=10⁴ → `output/ws1/8b/<tag>/`.
- "As-is" order is row index, i.e. **k-means cluster id within a month**. This is arbitrary but deterministic, not random.
- 42/589 groups have more than one centroid in some month (68 extra rows; at most 10 in one group). Only 6 have duplicates in the first or last month.
- Across the 42 affected groups: as-is vs random tie-breaking (3 seeds) gives 0–1 significance flips at 0.05, the same as rerunning as-is with a new MC seed. Month-averaging gives Λ 1 flip, δ² 4 flips, and drops group 323 (< 5 months). Totals across all groups: 59–62 significant on Λ under every rule.
- **Decision (user): keep as-is.** Report random tie-breaking and averaging as robustness checks in the SI. Text: "centroids from the same month are ordered arbitrarily (by cluster index); results are unchanged under random tie-breaking or within-month averaging."

*Step 3 (2026-09-22): corrected Λ.* `drift.lam_batch` re-estimates Σ̂ for every ordering. `drift.perm_test` uses MC with p = (1+b)/(B+1). **Decision (user): MC everywhere** (no exact enumeration, for consistency). Sanity checks: it matches `lambda_published` when Σ̂ is full rank. Rejection at α=0.05 on i.i.d. data is 0.05–0.06 for (T, r) ∈ {(6,3), (10,3), (30,3), (10,5)} (400 reps).
- **Grid** (as-is, B=10⁴, seed 0): basis ∈ {global PCA, group's own PCA}, r ∈ {2,3,5,10,20}, fixed r (test only T−2 ≥ r) or adaptive r_k = min(r, T−2). Groups with p < 0.05:

  | r | global fixed | global adaptive | group fixed | group adaptive | n tested (fixed) |
  |---|---|---|---|---|---|
  | 2 | 49 | 49 | 110 | 110 | 589 |
  | 3 | 60 | 60 | 119 | 119 | 589 |
  | 5 | 55 | 65 | 112 | 131 | 376 |
  | 10 | 44 | 74 | 69 | 113 | 129 |
  | 20 | 18 | 56 | 27 | 93 | 32 |

  δ² (full space): 37/589. Inclusive published p: 92/589.
- **Group's own PCA** (the SVD of the group's centred centroids, which doesn't depend on order, so it's valid under the null) has about 2× the power of global PCA. The global top axes capture between-topic spread rather than a group's own movement.
- **The significant set is fragile in r.** Jaccard of the p < 0.05 sets: group r=3 vs adaptive r_max=5 is 0.45; vs 10 it's 0.28; adaptive 10 vs 20 is 0.61. Agreement with δ² is 0.07–0.18. Named examples, group basis (p at r=3 / adaptive r_max=10 / 20):
  - religion (811, Fig 4c "significant"): 0.65 / 0.06 / 0.0001
  - laughing (774, Fig 4b "not significant"): 0.005 / 0.17 / 0.002
  - music (244, Table 1 "stable"): 0.018 at every r (T=5)
  - scientific (798): 0.02 / 0.0001 / 0.31
  - government (949): 0.013 / 0.15 / 0.02
- **Open (user):** how to handle the r-dependence. Options: (a) fix group-basis r=3, which tests all 589 groups, is simplest, and gives 119 significant; (b) adaptive with a stated r_max; (c) remove the choice with a max-over-r statistic (min p across r ∈ {2,3,5,10} under the same permutations, calibrated by the permutation distribution; still Λ, still exact under exchangeability). Step 6 (calibration on the km/kmp* nulls) could also rule settings out.
**LOCKED PROCEDURE (user, 2026-09-23).** Λ on the group's own top-r_k principal axes, r_k = min(5, ⌊(T−2)/2⌋); members in row order ("as-is"); Σ̂ re-estimated inside every shuffle; MC with B = 10⁴ and p = (1+b)/(B+1); seed 0; groups with T ≥ 5 and label ≠ −1. Command: `8b_drift_test.py --within_month asis --basis group --r_mode half --r 5 --seed 0` → `output/ws1/8b/asis_group_half_r5_s0/`. **155/589 groups at p < 0.05, 57 at p < 0.01.** Figure: [`figs/R1.1_driftmap_optionb_draft.png`](figs/R1.1_driftmap_optionb_draft.png).

*Step 4 results (2026-09-23): δ² comparison.* Same shuffles and settings, δ² = ‖c_last − c_first‖² in the full 384-d space (`p_d2` in the same CSV).
- **37/589 significant at p < 0.05** (2 at p < 0.01), vs 155 for Λ. Overlap: 24 groups; Jaccard 0.14; Spearman of the two p-value columns 0.68.
- δ² is much weaker here because its null is only over endpoint pairs, so p ≥ 2/(T(T−1)): no group with T ≤ 6 can reach 0.05. Λ uses the whole ordering.
- 13 groups are δ²-only. They have larger displacement (median 0.23 vs 0.20) but no consistent direction, which is the intended contrast: δ² sees net displacement, Λ sees a consistent mean step.
- Report δ² as a comparison column in the revised Table 1, as agreed.

*Step 6 results (2026-09-23): calibration on the real single-month null runs.* `src/8e_null_calibration.py --seed 0` → `output/ws1/8e_nullcal/`. These runs recluster one month repeatedly and align the results, so the resample index replaces time and the centroids are exchangeable by construction.
- **Across the 12 valid runs (1405 groups; 2007-02/04/05/06/07, 2010-02, 2015-01, 2017-02/03/04, 2020-06, 2021-06): Λ rejects 4.8% at α=0.05 (95% CI ±1.1) and 0.78% at α=0.01.** δ²: 4.0% and 0.43%. Flat across T bands (3.6–6.0%). **The test is calibrated on real data**, which answers R2.8's "false-positive rate of the WS1 test" for single-month nulls.
- **Three runs are not valid nulls and are excluded** (diagnosed, not assumed):
  - `km/kmp/2017-01` (100% rejection) and `km/kmp50/2007-03` (99%) are **warm-started chains**: each resample is initialised from the previous one, so adjacent resamples are ~3× closer than random pairs (ratio 0.31 vs ≈1.0 for the others) and the mean distance grows linearly with lag. They are drifting sequences, not exchangeable samples, so rejecting is correct behaviour.
  - `km/kmp/2007-04` reproduces the **identical** solution every resample (all pairwise distances exactly 0). Λ was then 0/0 and returned numerical noise (48% rejection).
- **Bug fixed in `drift.py` (affects other workstreams):** `perm_test` now returns (0.0, p = 1.0) when a group's centroids are numerically identical (`_degenerate`, tol 1e-10). Before the fix, degenerate groups produced meaningless p-values. No group in the published run is degenerate, so Table 1 is unaffected.
- This also supports the S2 claim: under a valid resampling null the pipeline finds essentially no drifting groups at α=0.01.

*Step 7 results (2026-09-23): multiple testing and step diagnostics.* `src/8f_diagnostics.py --seed 0` → `output/ws1/8f_diagnostics/groups.csv` (adds `q_lam`, `q_d2`, `rho1`, `rho1_z`, Shapiro columns), table [`tables/R2.6_fdr_counts_draft.md`](tables/R2.6_fdr_counts_draft.md), figure [`figs/R2.5_step_diagnostics_draft.png`](figs/R2.5_step_diagnostics_draft.png).
- **Final run uses B = 10⁵** (`output/ws1/8b/asis_group_half_r5_s0_B100k/`, 237 s). With B = 10⁴ the smallest attainable p (1.0×10⁻⁴) is above BH's tightest threshold (0.05/589 = 8.5×10⁻⁵), which cost us every q < 0.01 discovery (0 vs 6). Only 2 groups now sit at the 10⁻⁵ floor. At B = 10⁵: 157 groups at p < 0.05, 58 at p < 0.01.
- **R2.6 — BH-FDR across all 589 tested groups is severe:**

  | statistic | p<0.05 | p<0.01 | q<0.05 | q<0.01 |
  |---|---|---|---|---|
  | Λ | 157 | 58 | **22** | **6** |
  | δ² | 36 | 1 | 0 | 0 |

- **The FDR-surviving set is mostly platform, format and language drift, not the paper's headline topics.** Of the 22: "picture image jpg", "photo pic imgur", "youtube link hilarious", "karma posts subreddit", "pm messaged steamcommunity", "number numbers math", "spelling english word", "huh uh um", and two non-English groups ("som de nem", "en de se"). Substantive ones: "clinton hillary trump" (q=0.014), "pay jobs income" (q=0.021), "account money customer" (q=0.021), "english europe language" (q=0.036), "delicious tasty eat" (q=0.016), "games played gameplay" (q=0.003), "music listen track" (q=0.003), "wear wearing shirt" (q=0.048).
- **None of the named Table 1 / Fig 4 topics survive FDR:** president q=0.069, government q=0.10, laughing q=0.10, scientific q=0.13, music q=0.24, conspiracy q=0.27, republican q=0.48, religion q=0.59. **WS4: §3.2, the abstract and the Fig 4 narrative all need to be rewritten around this.**
- **R2.5 — step diagnostics** (descriptive; the permutation test needs neither Gaussianity nor a random walk):
  - **Lag-1 autocorrelation of steps: median −0.40, IQR [−0.46, −0.32]; 79% of groups below −0.3.** This is the MA(1) signature of *independent noise around a location* (theoretical −0.5), **not** a random walk with independent increments (which gives 0). Direct evidence against the random-walk framing in §2.6/S3.
  - Each group's ρ₁ is also z-scored against its own shuffle null (differencing any fixed point set gives negative ρ₁, so the null is not 0): median z = 0.25, i.e. the observed ordering is unremarkable versus exchangeable orderings. Consistent with the exchangeability framing.
  - Normality: pooling all group × dimension Shapiro–Wilk tests on the projected steps, **8.6% reject at 0.05** versus 5% expected; the p-value histogram is close to uniform. Approximate normality is reasonable, though the test does not rely on it.

- *Exploration of (b), 2026-09-23.* `8b_drift_test.py --basis group --r_mode {adaptive,half} --r <r_max>`; map from `8d_sig_map.py` → [`figs/R1.1_driftmap_optionb_draft.png`](figs/R1.1_driftmap_optionb_draft.png).
  - **The plain adaptive rule r_k = min(r_max, T−2) is bad.** It saturates the covariance: T−1 steps estimate a (T−2)-dim Σ̂, so Λ blows up (median 5.1 and max 2×10⁶ for saturated groups, vs median 1.4 otherwise) and **power collapses** — 9% significant among saturated groups vs 35% among the rest. It also makes the sig set swing with r_max (Jaccard 0.28–0.45).
  - **Fix: `--r_mode half`, r_k = min(r_max, ⌊(T−2)/2⌋)** — at least two steps per estimated dimension. Λ stays sane (max 8.9), power rises, and the r_max choice stops mattering much (Jaccard 0.72–0.87 across r_max 3/5/10).
  - Significant at p < 0.05: 136 (r_max=3), **155 (r_max=5)**, 163 (r_max=10), vs 119/131/113 unguarded and 145 for min-p. Seed 1 gives 153 (vs 155), so MC noise is ~2 groups.
  - i.i.d. calibration for the half rule: 0.045–0.058 at α=0.05 (T=5–30).
  - r_k actually used at r_max=5: r=1 for 130 groups, 2 for 159, 3 for 100, 4 for 71, 5 for 129. Sig rate by T: 14% (T ≤ 6), 24%, 37%, 45%, 56% (T > 30).
  - Named examples (r_max=5): president 0.003, government 0.010, laughing 0.009, scientific 0.016, conspiracy 0.11, republican 0.31, religion 0.43, music 0.085, team 0.89, woman 0.53.
  - **Recommended: option (b) with the half guard and r_max = 5.** One sentence in S3 ("Λ in the group's own top-r_k principal axes, r_k = min(5, ⌊(T−2)/2⌋)"), no min-p machinery, and slightly more power than (c).
- *Exploration of (c), not adopted (2026-09-22).* `drift.perm_test_minp` (Westfall–Young min-p over r ∈ {2,3,5,10} with r ≤ T−2, group basis, as-is, B=10⁴). Driver `src/8c_minp_map.py --seed 0` → `output/ws1/8c_minp/groups.csv`; map [`figs/R1.1_minp_map_draft.png`](figs/R1.1_minp_map_draft.png) (+ .pdf).
  - i.i.d. calibration: 0.050–0.058 at α=0.05 (T=5–40).
  - **145/589 significant** (66 at p < 0.01). Every one is also significant at some single r (219 are at ≥ 1 r; 48 at every applicable r). Jaccard vs single r=3: 0.55.
  - Rate by T: 8% (T ≤ 6), 22%, 30%, 60%, 69% (T > 30). Groups born in 2006 are over-represented (46/145), but the rate within each T band matches the other groups, so this is a T effect.
  - Named examples: scientific 0.0004, government 0.010, president (1001) < 0.001, laughing 0.017, music 0.018, religion 0.10, republican 0.31, conspiracy 0.24, team 0.91.

*Text changes needed* (draft under Manuscript changes once the numbers exist):
- **Abstract, §2.6, §3.2, S3:** reframe the null as temporal exchangeability. Replace "random walk null", "systematic directional drift" and "persistent, directional evolution" with wording like "displacement larger than expected if the topic's monthly centroids were exchangeable in time".
- **S3:** describe the actual permutation (shuffling centroid time order, not Δ_t), the telescoping identity, the per-shuffle covariance, the choice of r, and the p-value formula.
- **Keep the Gaussian LRT derivation only as motivation for the statistic.** Validity rests on exchangeability alone.
- **Add a limitation:** the test does not separate directional drift from diffusion.
- **Response to R1:** the reviewer is correct that shuffling Δ_t would leave Λ unchanged. The text misdescribed the procedure, which shuffles centroids. Say that we corrected the description, fixed a covariance-estimation issue we found while doing so, and that the numbers were updated.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R1.2

> Please clarify whether the MiniLM embeddings are L2-normalized before k-means, centroid-distance calculation, and drift analysis. Euclidean distance on unnormalized sentence embeddings can be affected by vector norm, so this small detail matters here. The related work on semantic representation could be slightly broadened. Studies such as ROUGE-SEM: Better evaluation of summarization using ROUGE combined with semantics, From coarse to fine: Enhancing multi-document summarization with multi-granularity relationship-based extractor, and Towards Curriculum Learning of Multi-Document Summarization Using Difficulty-Aware Mixture-of-Experts may provide some broader perspectives on semantic-aware evaluation and representation learning. These works are not direct topic-drift baselines, of course, but may be useful as methodological background.

**[internal]** Workstream: WS3 (normalization) / WS4 (citations) · Status: running (quick checks done 2026-09-22; test on re-normalized centroids waits for WS1)  
Plan: Normalization: all-MiniLM-L6-v2 ends in a Normalize module, so embeddings should be unit-norm. Verify on a stored zarr chunk (norms ≈ 1 up to fp16) and from the model's module list. Centroids are *not* unit-norm; check that the main results hold with re-normalized centroids / cosine distance (cheap, uses the WS1 statistic). WS3 quick checks alongside this: the model's `max_seq_length` (truncation discrepancy), and `[removed]`/`[deleted]` artifact clusters via the TF-IDF keywords in `mbkm_50/tfidf`. Citations: see E6.

*Step 1 results (2026-09-22).* `src/10a_quickchecks.py --seed 0` → `output/ws3/10a/{results.json,clusters.csv}` (clusters.csv has per (month, cluster) n, centroid norm, distances to the artifact vectors, the "removed"/"deleted" tf-idf score, top keywords).
- **Embeddings are L2-normalized.** The model's module list is `Transformer → Pooling (mean) → Normalize` (`modules.json`, and the loaded `SentenceTransformer`). Stored norms: 2006-06 zarr (all 20,888 rows, float32), 2008-06 npz (all 387,632, **float16**), 2012-06 npz and 2018-06 zarr (first 400k rows each) give norms in [0.99947, 1.00056], with none off by > 10⁻³. That's fp16 rounding; `1_embed.py` encodes in fp16. Note that npz months (2007-01 … 2015-10) store float16 and zarr months store float32.
- **Centroids are not unit-norm**: median 0.40, IQR 0.37–0.43, range 0.25–0.98. So k-means on unit vectors is spherical-ish k-means up to the centroid norm, and Euclidean distance between centroids ≠ cosine distance. Norms rise slightly over time (yearly median 0.37–0.38 in 2008–09 to 0.42 in 2022). High norms (> 0.6) flag near-duplicate spam clusters (see Cross-cutting issues).
- **Re-normalized centroids, δ_k only** (589 tested groups, row-index order, endpoint displacement): Spearman ρ = 0.956 between raw and re-normalized δ_k. Top-k overlap 5/10, 34/50, 79/100. The overall ranking is robust but the extreme top shifts, which matters because Table 1 is a top/bottom list. **To do once `src/drift.py` is final:** rerun the WS1 test on re-normalized centroids and report the agreement of the significant set.
- Draft response text: embeddings are unit-norm (the encoder ends in an L2 normalization layer; verified on stored vectors). So for embeddings, squared Euclidean distance = 2 − 2·cosine, and k-means is equivalent to maximizing cosine similarity to the (unnormalized) mean. Centroids are means of unit vectors (norm < 1). We report that results hold with re-normalized centroids.

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

**[internal]** Workstream: WS3 · Status: designing  
Plan (discussed 2026-09-22): **fix the topics, change only the geometry.** Don't re-run the pipeline.
1. Sample ~500 random member comments per (month, cluster) from `mbkm_50/labels`, for a subset of months. Default: semiannual (Jun and Dec), ~34 months; restrict further to months that contain tested groups if that helps.
2. Retrieve the text with one CPU pass per month over the raw dump on cm (Indexed Job over months).
   - Months ≤ 2015-10: match by comment ID from the metadata files. Exact.
   - Later months: replicate the `1_embed.py` filter and row order exactly.
3. **Alignment check first:** re-embed a few hundred retrieved comments with MiniLM and confirm cosine ≈ 1 with the stored rows. Do this for a metadata month and a non-metadata month before launching the full job.
4. Re-embed the samples locally on the 5090 with 1–2 other models. Proposed: all-mpnet-base-v2, plus one model from a different family such as gte-base or bge-base. Recompute centroids in each space. Keep the MiniLM topic-group assignments.
5. Compare across models:
   - rank correlation of per-group displacement δ_k
   - agreement of the WS1 test's significant set (once `src/drift.py` exists)
   - sign agreement of Fig 5 inter-topic distance changes
6. Optional add-on: for a few months, cluster the model-B embeddings and compare the partition with MiniLM's (ARI).

State in the response that this tests geometry, not cluster discovery. Decisions still open for the user: the month set, the per-cluster sample size, and the choice of models.

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

**[internal]** Workstream: WS3 · Status: designing  
Plan (agreed 2026-09-22): **don't downsample the whole corpus to the smallest month**; that would test a different estimator. The real concern is how centroid noise varies with volume. There's also a WS1 link: noisy early centroids break the exchangeability the permutation test assumes.
1. **Noise floor.** Per (month, cluster), estimate the centroid standard error √(tr Σ_within / n). Counts come from `labels`; within-cluster variance comes from a modest sample of embeddings per month (compute on cm if the sshfs reads are too slow). Plot it against month and compare with observed step sizes and δ_k.
2. **Low-volume exclusion.** Drop months below a threshold (e.g. before 2008 or under 1M comments), re-align, and re-run the WS1 test. Centroid-only and cheap; can start with δ_k before `src/drift.py` is ready.
3. **Optional high-end cap.** Recluster months above the cap (e.g. 5M) on a subsample; check that late-year groups and drift persist.
4. Also check whether significant groups are concentrated among groups whose first member falls in a low-volume month.

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

**[internal]** Workstream: WS1 · Status: drafted (numbers ready 2026-09-23; see R1.1 step 7)  
Numbers: ρ₁ median −0.40 (IQR [−0.46, −0.32]), 79% of groups below −0.3, versus −0.5 for independent noise and 0 for a random walk; z vs each group's own shuffle null median 0.25; Shapiro–Wilk on projected steps rejects 8.6% of group × dimension tests at 0.05 (5% expected). Figure: [`figs/R2.5_step_diagnostics_draft.png`](figs/R2.5_step_diagnostics_draft.png).
Plan: The permutation test does not assume Gaussianity or a random walk: its null is temporal exchangeability (see the R1.1 decision), and the Gaussian LRT form only motivates Λ. The δ² comparison uses no distributional form at all. Report diagnostics: lag-1 autocorrelation of steps (the MA(1) signature of independent clustering noise ≈ −0.5), normality of PCA-projected steps. Frame the test as an exchangeability null. Joint with R1.1.

**Response.** _TODO_

**Manuscript changes.** _none yet_

**Figures / tables.** _none yet_

### R2.6

> The analysis produces more than 1,000 aligned topic groups and reports significance using p<0.05. A multiple-testing correction such as FDR should be considered before identifying significantly drifting topics.

**[internal]** Workstream: WS1 · Status: drafted (numbers ready 2026-09-23; see R1.1 step 7)  
Numbers (B = 10⁵, 589 tested groups): Λ 157 at p<0.05 → **22 at q<0.05, 6 at q<0.01**; δ² 36 → 0 at q<0.05. Table: [`tables/R2.6_fdr_counts_draft.md`](tables/R2.6_fdr_counts_draft.md). B must be ≥ 10⁵: at 10⁴ the p-value floor sits above BH's tightest threshold and zeroes out the q<0.01 set.
**None of the currently named topics survive FDR** (president q=0.069, government q=0.10, scientific q=0.13, religion q=0.59) — WS4 needs this for §3.2 and the abstract.
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

**[internal]** Workstream: WS3 · Status: designing (clustering runs can start now; final test waits for WS1)  
Plan (agreed 2026-09-22):
- **Now:** fix the resampling so it is a true bootstrap or random-subsample resampling. Current `5a_dist` resamples differ only by k-means++ init, because minibatch updates are order-invariant within a batch. Write this as a new WS3 script; don't edit `5a_dist`. Choose early / middle / late months (e.g. 2009-06, 2015-06, 2021-06), write the manifest, and launch the clustering and alignment runs on cm. None of this depends on the test statistic.
- **After WS1 is complete:** apply the corrected WS1 test (`src/drift.py`) to each null run. Report the false-positive rate at α = 0.05, before and after BH correction, per window, and rewrite S2 (including the "resampled with replacement" wording).

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

**2026-09-23 (WS1 → all).** `src/drift.py` is ready for other workstreams. Locked call:
`drift.perm_test(drift.group_basis(X, max(1, min(5, (T-2)//2))), drift.lam_batch, n_perm=10000, rng=<seed>)`,
where `X = drift.trajectory(C, drift.group_rows(labels, coi, k)[0])`. It returns `(Lambda, p)`.
Two things to know: (1) `perm_test` now guards groups whose centroids are numerically identical
(returns p = 1) — this came up in the null runs, not in the published data; (2) do **not** use
`r_k = T-2`: it saturates the covariance and destroys power (see R1.1 step 3).
Calibrated at 4.8% on the single-month null runs (R1.1 step 6), which also covers part of R2.8.

**2026-09-23 (WS1 → WS4, important; corrected 2026-10-14).** After BH-FDR across the 589 tested groups, **22 groups have q < 0.05 and 6 have q < 0.01**. **None of the groups named in the published Table 1 or Fig 4 survive** (president q=0.069, government q=0.10, scientific q=0.13, laughing q=0.10, religion q=0.59, conspiracy q=0.27, oil q=0.32, republican q=0.48).
**But politics is not absent from the surviving set** — an earlier version of this note said so and was wrong. Among the 22: **1002 "clinton hillary trump" q=0.014**, **983 "himself political paul"** (Ron Paul) **q=0.021**, 821 "pay jobs income" q=0.021, 856 "account money customer" q=0.021, 991 "english europe language" q=0.036. The rest are platform/format/language groups (image and link sharing, karma/subreddit talk, spelling, non-English).
So the claim "politically contentious topics drift" is still supportable — it has to be **re-anchored to the groups that actually survive** rather than the ones originally chosen, and the examples in Table 1 and Fig 4 have to change. Alternative framing to consider: report at q < 0.10 as well (67 groups), where several of the originally named topics return. Also: the step diagnostics (R2.5) show lag-1 autocorrelation ≈ −0.4, the signature of noise around a location rather than a random walk, so "random walk" should leave the Methods framing too.
