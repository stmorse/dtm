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
| **WS1** Drift test | R1.1, R2.5, R2.6, Fig 5 test for R2.9 | `src/drift.py` (reusable statistic), `src/8_*.py`; revised Table 1, Fig 4, (Fig 5); rewritten §2.6 and S3 | — (**start first; others consume `src/drift.py`**) | **paused 2026-10-19, at a clean stopping point.** Steps 1–8 done; R1.1, R2.5, R2.6 responses and manuscript changes drafted; all tables and figures produced and reproducible from `src/8*.py`. Headline: **157 of 589 groups at p<0.05, 22 at q<0.05, 6 at q<0.01; none of the published Table 1 / Fig 4 topics survive** (best is president, q=0.069). **WS4 needs:** R2.6 (what to say about the topic change), R1.1 step 8 (wording for 1002 and the 221/244 naming trap), R2.5 (the random-walk framing has to go). **Open, neither blocking:** (a) Fig 5 test for R2.9 — WS1 recommends a wording fix instead, see the 2026-10-14 note; (b) `straightness_z` as a CSV column, not built. **To resume:** everything runs from `src/`, `8g_group_table.py` → `8h_drift_map.py` in that order, ~10 s each. |
| **WS2** Alignment & k sensitivity | R2.4, R2.2, R1.3 | `src/9_*.py`; SI section and figures | WS1 statistic (can start on partition stability without it) | todo |
| **WS3** Scale & model robustness | R2.3, R2.1, R2.8, R1.2 (normalization) | `src/10_*.py`, `manifests/rev-*.yml`; subsample dataset on sciclone; SI sections | WS1 statistic; cm time (start the cm jobs early) | running: step 1 (quick checks) done 2026-09-22. R2.1 design agreed 2026-10-19 (three arms, 45M sample, all on cm CPU) — see R2.1. Building: model pre-stage + test drive first, then the sampler and the retrieval Job. |
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

**[internal]** Workstream: WS1 · Status: steps 1–8 done, responses drafted, Table 1 + Fig 4 produced (2026-10-14)

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

*Step 8 (2026-10-14): revised Table 1 and Fig 4.* **Decision (user): report the groups that are significant under BH-FDR, even where that changes the table.** Two scripts, so curation is done once on the table and the figure follows it:
- `src/8g_group_table.py` joins the locked B=10⁵ run with the step diagnostics, adds TF-IDF labels, lifespan and path-shape columns, applies BH over all tested groups, and writes `tables/R1.1_groups_all_draft.csv` (every group, every column), `tables/R1.1_table1_draft.{csv,md}` (the q < 0.05 set) and `tables/R1.1_table1_published_draft.{csv,md}` (what became of the published Table 1 / Fig 4 topics). `--exclude` drops curated-out artifacts from Table 1 and records the count.
- `src/8h_drift_map.py` draws the revised Fig 4 from that CSV. `--annotate auto:N | list: | file: | published | none` controls which groups are named; `--exclude` draws them in a third colour, removes them from the significant count, and states the number on the figure so a note in the text can match it. `--traj coi,coi` adds Fig 4b/4c-style trajectory panels (group's own top-2 PCs, coloured by month).
- New in `drift.py` (additive, safe for other workstreams): `bh_qvalues` (now the single BH implementation; `8f` imports it), `path_length`, `straightness`, `max_step`, `load_keywords`, `group_label`.
- **`straightness` = δ / Σ‖Δₜ‖** is a new descriptive column, not a test: 1 is a straight line, ≈1/√(T−1) a random walk, ≈0 noise around a location. It is the column that speaks to whether a surviving group looks like travel or like jitter, and it is worth reporting next to Λ. `max_step` flags one-off level shifts (e.g. a 2016 jump) as opposed to gradual movement.
- **Run 2026-10-14** (9 s): `8g_group_table.py --seed 0`, then `8h_drift_map.py --seed 0 --annotate list:53,573,712,821,1002`. Reproduces step 7 exactly — 589 tested, 157 at p<0.05, 58 at p<0.01, **22 at q<0.05, 6 at q<0.01**; δ² 0 at q<0.05.
- **The revised Table 1** (`tables/R1.1_table1_draft.md`), q ascending: 4 "som de nem", 53 "number numbers math", 221 "music listen track", 695 "games played gameplay", 826 "bang dun holla", 573 "picture image jpg", 1002 "clinton hillary trump", 330 "delicious tasty eat", 821 "pay jobs income", 856 "account money customer", 983 "himself political paul", 78 "pm messaged steamcommunity", 491 "huh uh um", 991 "english europe language", 22 "en de se", 650 "youtube link hilarious", 563 "karma posts subreddit", 646 "photo pic imgur", 712 "spelling english word", 428 "wear wearing shirt", 478 "herself bitch her", 655 "greatest hilarious incredible".
- **The published topics** (`tables/R1.1_table1_published_draft.md`): best is president (1001) at q=0.069, rank 30 of 589; then government and laughing 0.10, scientific 0.13, delicious 0.16, music 0.24, conspiracy 0.27, oil 0.32, republican 0.48, games 0.49, religion 0.59, woman 0.67, team 0.93. Note **221 "music listen track" (q=0.003) is significant while the published "music" group 244 (q=0.24) is not** — they are different aligned groups (T=18 vs T=5), so the revised text must not say "music drifts" or "music is stable" without naming the group.
- **Power is governed by T, and FDR survival needs T ≥ 9.** Significance rate by T quartile: 0% (T≤6, n=213), 0% (7–8, n=134), 4.4% (9–11, n=113), 13.2% (12–83, n=129). Coverage (months observed / months spanned) is *not* a confound: within the longest-T quartile the rate is 13.8% for sparse groups vs 12.5% for dense ones. So the honest statement is that the test cannot detect drift in short-lived groups, not that short-lived groups are stable.
- **Path shape: the surviving trajectories are not straight, but their ordering is not arbitrary either.** `straightness` = δ/Σ‖Δₜ‖ is 0.03–0.32 for the 22, in every case below the random-walk reference 1/√(T−1) — month-to-month noise dominates each step, consistent with ρ₁ ≈ −0.4 (R2.5). But compared with *shuffled orderings of the same centroids* (B=500, descriptive only, not a new test), the 22 are a median **1.7× straighter than their own shuffle null** (median z = 2.15; 55% have z > 2) against 1.2× and 11% for the other 567. So the chronological ordering does carry directional information; the absolute geometry is just noisy. **Not in any script**: computed in a throwaway, unlike the gap analysis below. Say so and it becomes a `straightness_z` column in `8f`.
- **1002 "clinton hillary trump" moves in jumps between election cycles — and that supports the paper's claim rather than undercutting it (user, 2026-10-14).** The group is present in 14 months: 2008-01, 2008-05, eleven consecutive months 2016-02 to 2016-12, then 2020-03 and 2020-11. Within 2016 it is nearly stationary (1-month steps 0.03–0.13, median 0.07); the cross-cycle steps are 3–5× larger (0.20–0.33), and the largest, 2016-12 → 2020-03, is 0.333, exceeding the group's net displacement of 0.226. **The grouping was agnostic to time**: HDBSCAN placed these centroids in one group on embedding proximity alone, with no access to period. So the same embedding-recognised topic occupies measurably different regions in different election cycles, which is exactly the time-dependence the test is designed to detect. A jump is a form of time-dependence, and since the test cannot separate drift from diffusion for *any* group, "gradual" was never available. **Wording for WS4:** "occupies measurably different regions in different periods", not "drifts steadily" or "gradually". Same shape: 983 "himself political paul", 991 "english europe language" (max step 0.399 vs δ 0.262), 4 "som de nem" (0.401 vs 0.224).
- **Does displacement accumulate with elapsed calendar time?** Descriptive check over all 5,265 consecutive pairs (gap in months vs ‖Δ‖, same machinery as the step-6 warm-start diagnosis). **Corpus-wide, no:** median ‖Δ‖ is 0.139 at a 1-month gap and 0.147 beyond 24 months, a ratio of 1.06. Per-group Spearman(gap, ‖Δ‖), each group its own control: median 0.061, 55% positive (sign test p = 0.029). **Among the 22 survivors, modestly yes:** median 0.160, 77% positive. **For 1002, strongly: ρ = 0.858** (13 steps, driven by the within-cycle vs between-cycle contrast, so low power). Three consistent tiers — the test selects for groups where time matters, and a gap effect is one way for it to matter.
  - **Use for R2.5:** the corpus-wide flatness is a second line of evidence against the random-walk framing, independent of ρ₁ and more direct — a diffusing random walk would give E‖Δ‖² ∝ gap, and it does not. **Caveat to state:** monthly measurement noise puts a floor under ‖Δ‖, so flatness bounds the size of any drift rather than excluding it.
  - **Now in `8f_diagnostics.py`** (2026-10-19): `gap_steps`, `gap_rho`, `gap_table`; columns `gap_rho`, `gap_n` in the diagnostics and master CSVs; panel (c) of the R2.5 figure; table [`tables/R2.5_gap_profile_draft.md`](tables/R2.5_gap_profile_draft.md). Re-running `8f` reproduces every earlier ρ₁ and Shapiro number unchanged (ρ₁ median −0.4017, 79.1% below −0.3, pooled Shapiro 8.6%). **The quantified version of the diffusion comparison:** at a median gap of 37.5 months the observed median step is 0.147 against a diffusion prediction of 0.848 — a ratio of 0.17.
- **Limitation this exposes:** the test treats a group's centroids as an evenly spaced sequence, but T counts *months in which the group appears*, not calendar time; median coverage is 0.08, so most groups are sporadic and some steps span years. Worth stating in the limitations paragraph alongside the drift/diffusion point — though note the gap analysis above shows step size is nearly flat in gap corpus-wide, so the uneven spacing is not, in itself, driving the results.

**Response.** We thank the reviewer for this comment, which identified a real error.

The reviewer is correct: permuting the step vectors Δₜ leaves their mean and covariance unchanged, so Λ would be invariant and the test vacuous. Appendix S3 described the procedure incorrectly. The code permuted the time order of the monthly *centroids* and recomputed the mean step from the permuted sequence. We have rewritten S3 to describe the procedure that was actually run.

Examining the code in response to this comment, we found a second problem. Because the mean step telescopes, μ̄ = (c_last − c_first)/(T−1), the statistic computed on a permuted sequence depends only on which two centroids land at the ends; and the implementation held the covariance Σ̂ fixed at the value estimated from the observed ordering. The permutation distribution was therefore not a valid reference distribution for the statistic. We have corrected this: Σ̂ is re-estimated inside every permutation. Three further changes follow:

1. *Dimension.* With Σ̂ re-estimated per permutation it must be estimable. The original analysis projected onto 45 principal components fit on all 10,200 centroids, but 583 of the 589 tested groups have fewer than 47 monthly centroids, so Σ̂ was rank-deficient for nearly all of them. We now compute Λ in each group's own leading principal axes, with r_k = min(5, ⌊(T−2)/2⌋), i.e. at least two steps per estimated dimension. Using the group's own axes rather than global axes also matters: the global leading axes capture variation between topics rather than the movement of any one topic.
2. *p-value.* We use p = (1 + b)/(B + 1) with B = 10⁵ permutations. The original used #{null > observed}/B, which excludes the observed ordering and is anti-conservative; with the strict inequality, groups with as few as five centroids could be reported significant on the strength of tied values.
3. *Multiple testing.* See our response to Reviewer 2, comment 5 (R2.6).

We have also corrected the interpretation. The null hypothesis actually tested is that a topic group's monthly centroids are **exchangeable in time**. Rejection means the group's position in embedding space depends on when it is measured. It does not establish directional drift as opposed to accumulated diffusion: a driftless random walk also violates exchangeability, because the expected squared distance between two centroids grows with their separation in time and the first and last months are the most separated pair. We now state this, and have added it as an explicit limitation. The Gaussian likelihood-ratio derivation is retained only as motivation for the form of Λ; the validity of the test rests on exchangeability alone.

We verified the calibration of the corrected test on data rather than by assumption. The single-month resampling runs reported in S2 cluster the same month repeatedly, so the aligned centroids are exchangeable by construction and the test should reject at the nominal rate. Across 1,405 groups from twelve such runs it rejects 4.8% of the time at α = 0.05 and 0.78% at α = 0.01.

The corrected analysis gives materially different results, and we report them as such. Of 1,006 aligned topic groups, 589 have at least five monthly centroids and are tested. 157 reach p < 0.05; after Benjamini–Hochberg correction, 22 reach q < 0.05 and 6 reach q < 0.01. The groups named in the published Table 1 and Fig 4 are not among them (president q = 0.069, government q = 0.10, scientific q = 0.13, religion q = 0.59). We have rebuilt Table 1 and Fig 4 around the groups that do survive, revised §3.2 and the abstract accordingly, and added a table showing what became of the previously named topics so that the change is visible to readers. We also report the raw displacement δ_k and a second permutation test on δ² = ‖c_last − c_first‖² in the full embedding space alongside Λ, so that the dependence of the conclusions on the choice of statistic can be seen directly.

**Manuscript changes.**
- **§2.6 and S3.** Replace the description of the permutation test. State that the null is temporal exchangeability of a group's monthly centroids; that the permutation shuffles the time order of the centroids (not the steps Δₜ); that μ̄ telescopes to (c_last − c_first)/(T−1); that Σ̂ is re-estimated within each permutation; that Λ is computed in the group's own top-r_k principal axes with r_k = min(5, ⌊(T−2)/2⌋); and that p = (1+b)/(B+1) with B = 10⁵. Keep the Gaussian LRT derivation, but label it as motivation for the form of Λ rather than as an assumption.
- **§2.6, §3.2, abstract.** Remove "random walk null", "systematic directional drift" and "persistent, directional evolution". Replace with wording of the form "displacement larger than would be expected if the topic's monthly centroids were exchangeable in time".
- **§4 (limitations).** Add: the test detects dependence of position on time; it does not separate directional drift from accumulated diffusion, since both violate exchangeability. Add also that δ_k depends on two noisy endpoints and grows with a group's lifespan.
- **Table 1.** Replace with the FDR-significant groups, with columns δ_k, Λ_k, p_k, q_k and the δ² p-value; say in the caption how groups were selected (q < 0.05 among 589 tested) and how many, if any, were excluded as platform artifacts. Add the companion table of previously named topics.
- **Fig 4.** Recolour by q rather than by Λ ≥ 5 and T ≥ 10, and fix the caption, which currently describes the colouring as "significant".
- **S2.** Note that the corrected test rejects 4.8% of the time on the single-month resampling nulls.

**Figures / tables.** Produced by `8g_group_table.py` and `8h_drift_map.py` (run 2026-10-14, see step 8):
- [`tables/R1.1_groups_all_draft.csv`](tables/R1.1_groups_all_draft.csv) — all 589 tested groups, all columns (supporting information).
- [`tables/R1.1_table1_draft.md`](tables/R1.1_table1_draft.md) — revised Table 1.
- [`tables/R1.1_table1_published_draft.md`](tables/R1.1_table1_published_draft.md) — the published Table 1 / Fig 4 topics under the corrected test.
- [`figs/R1.1_fig4_revised_draft.png`](figs/R1.1_fig4_revised_draft.png) (+ .pdf) — revised Fig 4, 22 of 589 coloured, five groups annotated (53, 573, 712, 821, 1002 — a draft selection spanning the two strands: platform/format/language and substantive).
- Superseded drafts: [`figs/R1.1_driftmap_optionb_draft.png`](figs/R1.1_driftmap_optionb_draft.png), [`figs/R1.1_minp_map_draft.png`](figs/R1.1_minp_map_draft.png).

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

**[internal]** Workstream: WS3 · Status: designing → building (design agreed with user 2026-10-19)

**Design: three arms off one retrieval pass.** The September plan (fix the topics, change only the geometry) survives only as arm B. The user's objection on 2026-10-19 was correct: a MiniLM-conditioned comparison can say whether fixed comment sets move the same way, but it cannot say whether model B would have found the structure at all. So the independent re-run (A) is now the primary arm.

| Arm | Question | Partitions | Groups | Bias |
|---|---|---|---|---|
| **A** | Does the month-level topic structure reproduce under another encoder? | k-means re-run in each space on the same sampled comments | UMAP+HDBSCAN re-run on model-B centroids | none — model B never sees a MiniLM-selected input |
| **B1** | Does the measured movement of *the paper's* topics reproduce? | MiniLM's, inherited | MiniLM's, inherited | MiniLM-defined topics by construction; stated as such |
| **B2** | Does the *grouping* reproduce, holding the clustering fixed? | MiniLM's, inherited | UMAP+HDBSCAN re-run on the arm-B centroid matrix | none beyond the fixed partition |

*Why B needs no new alignment.* `align_model_HDBSCAN_labels.npz` labels **(month, cluster) pairs** (row = period·50 + cluster), not embeddings. A group is a list of addresses, e.g. group 1002 = {(2008-01, c_a), (2008-05, c_b), (2016-02, c_c), …} (14 pairs). Arm B1 looks up each address's model-B centroid, stacks them in month order, and runs the WS1 statistic on the 768-d trajectory. Same T, same months, same membership; only the coordinates change.

*Why B2 is the cleanest single test.* The arm-B centroid matrix (10,200 × 768) has **identical units** to the published one, so the two HDBSCAN partitions are directly comparable: ARI over 10,200 (month, cluster) pairs, no matching heuristic, no selection. It answers "do 1002's 14 members still land in one group?" One UMAP+HDBSCAN run, seconds of compute.

*Cross-model comparison is by membership, never by centroid vectors.* MiniLM is 384-d and the model-B encoders are 768-d, so centroid vectors are incomparable across spaces. Arm A compares partitions (ARI/AMI, matched-cluster member overlap); centroids are only ever compared within a space.

*The reference scale is within-model instability, not 1.0.* k-means with k=50 in 384 dimensions is not seed-stable, so arm A also runs several k-means seeds within MiniLM space on the same sample. The claim to aim for is "structure reproduces about as well across encoders as across restarts of our own method". This also feeds R2.4 and R2.8.

*Three partitions per month, not two.* MiniLM-full (published labels restricted to the sampled rows), MiniLM-sample (k-means re-run on the sample in MiniLM space), modelB-sample. Full-vs-sample isolates the **subset** effect; sample-vs-sample isolates the **model** effect. Without the middle one any disagreement is uninterpretable.

**Decisions (user, 2026-10-19).**
- **Sample:** union of a **uniform 200k/month** draw (arm A; preserves cluster proportions, which re-clustering requires) and a **stratified 2,000 per (month, cluster)** draw (arm B; uniform precision). ~250k/month, **~45M comments over all 204 months**. Random members — *not* nearest-to-centroid, which is what the TF-IDF step used and would bias every centroid inward.
- **Months:** all 204.
- **Models:** `all-mpnet-base-v2` (same training recipe, different architecture/width: tests capacity) **and** `BAAI/bge-base-en-v1.5` (RetroMAE + different fine-tuning data: tests the training recipe). Both in the same pass. `bge-m3` is cached locally if a multilingual third is ever wanted.
- **Compute: retrieval on cm (CPU), embedding on prospero (GPU).** This reverses an earlier call in this plan, which was based on a synthetic benchmark that overstated CPU throughput ~5×. Measured on **real dump text**, 2026-10-19:

  | | MiniLM | mpnet | bge-base |
  |---|---|---|---|
  | prospero CPU, 12 threads (sent/s) | 344 | 43 | 48 |
  | → min per 250k month | 12 | 98 | 87 |
  | prospero 5090, fp16 (sent/s) | 20,700 | 6,650 | 8,520 |
  | → min per 250k month | 0.2 | 0.6 | 0.5 |

  CPU embedding of both model-Bs would be ~630 pod-hours; the 5090 does all three models over the full 45M in **~4 hours** (MiniLM 38 min, mpnet 1.9 h, bge 1.5 h). The transfer that worried us is small: 45M bodies ≈ 11 GB raw, ~5 GB gzipped, and sshfs measured **~100 MB/s** (also 2026-10-19 — the slow step-1 reads were float64 distance passes in that script, not the mount), so minutes.
  So: **cm pods stream, sample, write gzipped text, and emit the verification certificate** (which needs MiniLM for only ~300 rows, i.e. seconds). **prospero embeds all three models on the 5090**, runs the per-month k-means, and does the alignment and drift analysis. No cm GPU is needed anywhere.
- **Input window.** All three encoders end in `Normalize`, so every space is unit-norm and the SE formula carries over. But their windows differ: MiniLM 256 tokens, mpnet 384, bge 512. **Proposal: cap all models at `max_seq_length = 256`** so model B sees exactly what MiniLM saw — we are testing the encoder, not the window — and note that 2.1% of comments are affected (step 1). It is also ~15% faster. *Open for the user.*
- **Manifest:** Indexed Job, completion index → month, parallelism cap set by the user.

**Sampling needs no metadata pass.** `labels_YYYY-MM.npz` has one label per row for **all 204 months** (verified 2026-10-19 against `totals.csv` for 2006-06, 2008-06, 2015-11, 2018-06, 2021-06, 2022-12). So the index sets are drawn locally and cheaply from `labels` + `totals.csv` before anything launches; only text retrieval needs the dump pass.

**Verification, per month, in-pod.**
- 2006-01 … 2015-10: **exact comment-ID match** against `metadata/` (gzip CSV, 2006–2012) or `metadata2/` (pkl, 2013 … 2015-10). No embedding read needed.
- 2015-11 … 2022-12: no metadata, but these months are all **zarr**, so read the chunks holding a few hundred sampled rows and confirm cosine ≈ 1 against re-embedded MiniLM.
- This split means the 10–30 GB npz months (2008–2014) never have to be loaded: stored embeddings are only needed for the cosine check, and those months are covered by the ID check instead. MiniLM embeddings for the *analysis* are recomputed in-pod from the retrieved text (~5 min/month).

**Statistical sufficiency of the 45M subset** (reviewer-facing, four ways):
1. Arm A compares two encoders *on the same comments*, so sampling is common to both and cannot favour either. Sufficiency only matters for the auxiliary claim that the subset recovers the full month's structure.
2. That claim is **measured, not assumed**: ARI(MiniLM-full, MiniLM-sample), which doubles as the yardstick for the cross-model number.
3. Analytic centroid precision: SE = √((1−‖c‖²)/n) ≈ √(0.84/n) since embeddings are unit-norm. Uniform sample gives median 3,900 members/cluster → SE 0.015 (1st pct 1,280 → 0.026); stratified 2,000 → 0.020. Displacement error √2× that = 0.02–0.04, i.e. 15–25% of WS1's median monthly step (0.139). Report as a noise band on δ.
4. **Split-half reliability** (δ on two disjoint halves in MiniLM space, Spearman–Brown to full n) plus a **subsample ladder** (every metric recomputed at 250/500/1,000/2,000 per cluster from the retrieved text) showing the curves plateau. Costs nothing extra and is the most persuasive form: an empirical convergence curve, not a power calculation.

**Comparisons to report.** Rank correlation of δ_k, Λ_k and p_k across all 589 tested groups (never conditioning on significance — selecting the q<0.05 set and asking whether it replicates is a winner's curse); per-group status for the named topics (1002, 983, 221, 821, 856, 991, plus the published Table 1 / Fig 4 topics); ARI tables for arms A and B2; and WS1's gap-profile diagnostic recomputed in each space, which should stay flat if flatness is a property of the discourse rather than of MiniLM.

**Scripts.** `src/ws3_common.py` (shared stream filter, sampling, row readers), `src/10b_model_check.py` (preliminary: model cache + test drive + throughput benchmark), `manifests/rev-ws3-dev.yml` (interactive CPU pod), then the sampler, the retrieval Job script, and the arm A/B analysis scripts.

*Preliminary results (2026-10-19), `10b_model_check.py` on prospero → `output/ws3/10b/results.json`.*
- **Encoders pre-staged** into the cm cache `/sciclone/proj-ds/geograd/stmorse/.cache/huggingface/hub` (419 MB each), written from prospero over sshfs, so **pods need no network**. All three load offline and end in `Normalize`.
- **The stream filter is exact.** 2006-06 streamed in full: 20,888 kept rows = `totals.csv` = `len(labels)`, index contiguous.
- **ID check: 300/300.** Streamed `id` at row *i* equals `metadata/metadata_2006-06.csv`'s id for `idx = i`.
- **Cosine check: median and min = 1.00000** (300 rows, 2006-06 zarr) between MiniLM re-embedded on CPU from retrieved text and the stored fp16 GPU rows. Retrieval-to-row alignment is therefore exact, and re-embedding is interchangeable with reading stored rows — which matters because reading 250k scattered rows from a late month's zarr would touch nearly every 850 MB chunk, whereas re-embedding is minutes. **The analysis re-embeds; stored rows are only used for this check.** State in the response that arm B tests geometry, not cluster discovery, and that arm A tests both.

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
1. **Noise floor.** Per (month, cluster), estimate the centroid standard error √(tr Σ_within / n). **Simplification found in step 1 (2026-10-19): this needs no embedding reads at all.** Embeddings are unit-norm, so for a centroid that is the mean of its members, tr Σ_within = 1 − ‖c_j‖² exactly; the standard error is √((1−‖c_j‖²)/n_j) with n_j from `labels`. Computable for all 10,200 clusters locally in seconds. Validate against a direct computation on two months (MBKM centroids are running means, so only approximately the member mean), then plot against month next to WS1's observed step sizes and δ_k.
2. **Low-volume exclusion.** Drop months below a threshold (e.g. before 2008 or under 1M comments), re-align, and re-run the WS1 test. Centroid-only and cheap; can start with δ_k before `src/drift.py` is ready.
3. **Optional high-end cap.** Recluster months above the cap (e.g. 5M) on a subsample; check that late-year groups and drift persist. **Now largely free (2026-10-19):** R2.1's arm A re-clusters a *fixed* 200k uniform sample for every month in MiniLM space, which is exactly this controlled-subsampling analysis. One retrieval pass serves R2.1 and this step; take the numbers from there rather than building a separate run.
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

**[internal]** Workstream: WS1 · Status: response drafted 2026-10-14 (numbers ready 2026-09-23; see R1.1 step 7)  
Numbers: ρ₁ median −0.40 (IQR [−0.46, −0.32]), 79% of groups below −0.3, versus −0.5 for independent noise and 0 for a random walk; z vs each group's own shuffle null median 0.25; Shapiro–Wilk on projected steps rejects 8.6% of group × dimension tests at 0.05 (5% expected). **Step size vs calendar gap (added 2026-10-19): flat — median step 0.139 at a 1-month gap vs 0.147 beyond 24 months (1.06×), where a random walk predicts 0.848 (ratio to diffusion 0.17); per-group Spearman median 0.061, 55% positive.** Figure: [`figs/R2.5_step_diagnostics_draft.png`](figs/R2.5_step_diagnostics_draft.png) (3 panels). Table: [`tables/R2.5_gap_profile_draft.md`](tables/R2.5_gap_profile_draft.md).
Plan: The permutation test does not assume Gaussianity or a random walk: its null is temporal exchangeability (see the R1.1 decision), and the Gaussian LRT form only motivates Λ. The δ² comparison uses no distributional form at all. Report diagnostics: lag-1 autocorrelation of steps (the MA(1) signature of independent clustering noise ≈ −0.5), normality of PCA-projected steps. Frame the test as an exchangeability null. Joint with R1.1.

**Response.** The test does not assume that the step vectors are i.i.d. Gaussian. Its null hypothesis is that a topic group's monthly centroids are exchangeable in time, and the p-value is computed from the permutation distribution of the statistic under that null, so no distributional form is required for its validity. The Gaussian likelihood-ratio argument motivates the particular quadratic form of Λ and nothing more; we have revised §2.6 and S3 to say so, and the companion test on δ² = ‖c_last − c_first‖² makes no distributional assumption at all. (The permutation procedure itself was also misdescribed in the original S3; see our response to Reviewer 1, comment 1.)

We agree that the behaviour of the steps is worth reporting, and have added diagnostics on all 589 tested groups (new supporting figure).

*Temporal dependence.* The lag-1 autocorrelation of the step vectors, ρ₁ = Σ⟨Δₜ, Δₜ₊₁⟩ / Σ‖Δₜ‖², has median −0.40 (IQR [−0.46, −0.32]), with 79% of groups below −0.3. The reference values are instructive. Differencing independent measurements of a fixed location gives first differences of the form εₜ − εₜ₋₁, a moving-average process of order one with ρ₁ = −0.5, because consecutive steps share a term with opposite sign. A random walk with independent increments gives ρ₁ = 0, and a random walk with non-zero drift gives a positive value under this (uncentred) definition, since every step then contains a common component. The observed values sit close to the independent-noise case and far from either random-walk case, in the direction opposite to the one drift would produce. We have therefore removed the random-walk description of the steps from §2.6 and S3 and replaced it with the exchangeability framing.

*Accumulation with elapsed time.* A second diagnostic tests the same question more directly. If the monthly centroids followed a diffusing random walk, the expected squared distance between two observations would grow with the time separating them; if they are independent measurements of a stable location, step size would be unrelated to that separation. Topic groups are observed in irregular months, so the 5,265 consecutive pairs in the data span gaps from one month to more than a decade, and the comparison is available without any additional modelling. The median step length is 0.139 for pairs one month apart and 0.147 for pairs more than two years apart, a ratio of 1.06; diffusion anchored on the one-month pairs would predict 0.848 at the longest gaps, so the observed steps are about a fifth of what a random walk implies. Taking each group as its own control, the median within-group Spearman correlation between gap and step length is 0.061, with 55% of groups positive. We note the limitation that month-to-month measurement noise places a floor under the observed step length, so this result bounds the magnitude of any systematic movement rather than excluding it.

*Against each group's own null.* Differencing any finite fixed set of points induces negative ρ₁ by construction, so we also compare each group's ρ₁ with the distribution obtained from permuted orderings of its own centroids: the median z-score is 0.25, i.e. the chronological ordering is unremarkable relative to reshuffled orderings of the same data.

*Normality.* Pooling Shapiro–Wilk tests over all group × projected-dimension combinations, 8.6% reject at the 0.05 level against 5% expected, and the p-value histogram is close to uniform. Approximate normality of the projected steps is a reasonable description, though, as above, the test does not depend on it.

**Manuscript changes.**
- **§2.6 / S3.** State that validity rests on exchangeability alone, and that the Gaussian LRT derivation motivates the form of Λ. Remove the claim that the steps are i.i.d. Gaussian and the random-walk description of them. (Shared with R1.1.)
- **New SI subsection "Step diagnostics".** Report ρ₁ (median, IQR, fraction below −0.3), the −0.5 / 0 / positive reference values and what each corresponds to, the z-scores against each group's own permutation null, the step-size-vs-gap profile with its diffusion reference, and the pooled Shapiro–Wilk result. Reference the new figure.

**Figures / tables.** [`figs/R2.5_step_diagnostics_draft.png`](figs/R2.5_step_diagnostics_draft.png) (+ .pdf) — (a) distribution of ρ₁ with the −0.5 and 0 reference lines, (b) pooled Shapiro–Wilk p-values against the uniform reference, (c) median step length by calendar gap against the random-walk and flat references. [`tables/R2.5_gap_profile_draft.md`](tables/R2.5_gap_profile_draft.md) (+ .csv) — the pooled gap profile. Columns `rho1`, `rho1_z`, `gap_rho`, `gap_n`, `shapiro_frac_lt05` in [`tables/R1.1_groups_all_draft.csv`](tables/R1.1_groups_all_draft.csv).

### R2.6

> The analysis produces more than 1,000 aligned topic groups and reports significance using p<0.05. A multiple-testing correction such as FDR should be considered before identifying significantly drifting topics.

**[internal]** Workstream: WS1 · Status: response drafted 2026-10-14 (numbers ready 2026-09-23; see R1.1 step 7)  
Numbers (B = 10⁵, 589 tested groups): Λ 157 at p<0.05 → **22 at q<0.05, 6 at q<0.01**; δ² 36 → 0 at q<0.05. Table: [`tables/R2.6_fdr_counts_draft.md`](tables/R2.6_fdr_counts_draft.md). B must be ≥ 10⁵: at 10⁴ the p-value floor sits above BH's tightest threshold and zeroes out the q<0.01 set.
**None of the currently named topics survive FDR** (president q=0.069, government q=0.10, scientific q=0.13, religion q=0.59) — WS4 needs this for §3.2 and the abstract.
Plan: Apply Benjamini–Hochberg FDR across all tested groups; report counts at q < 0.05 and q < 0.01; use q-values in Table 1 and Fig 4.

**Response.** We agree, and have applied the Benjamini–Hochberg procedure across all tested topic groups. Table 1 and Fig 4 now report q-values, and the text identifies drifting topics by q rather than by p.

Some clarification of the denominator: the alignment step produces 1,006 topic groups, but a group needs at least five monthly centroids for the test to be defined, which leaves 589. These 589 are the family over which the correction is applied, and we now state this in §2.6.

The effect is substantial. Under the corrected test (Reviewer 1, comment 1) 157 of the 589 groups reach p < 0.05 and 58 reach p < 0.01; after correction, 22 reach q < 0.05 and 6 reach q < 0.01. For the δ² comparison statistic, 36 groups reach p < 0.05 and none survive correction. We report the counts at both thresholds, and note in the text that at q < 0.05 about one of the 22 reported groups is expected to be a false discovery.

One implementation detail bears on reproducibility. The smallest attainable Monte-Carlo p-value is 1/(B+1), and this must fall below the smallest Benjamini–Hochberg threshold, α/m = 0.05/589 = 8.5 × 10⁻⁵. We therefore use B = 10⁵ permutations rather than the 10⁴ of the original analysis; at B = 10⁴ no group can reach q < 0.01 however strong its evidence. This is now stated in S3.

The correction changes which topics are reported. Most of the surviving groups describe changes in platform conventions, posting formats and language — image- and link-sharing vocabulary, subreddit and karma vocabulary, spelling, and two non-English groups — rather than the political topics emphasised in the original version. A smaller number are substantive, including a group with the keywords "clinton hillary trump" (q = 0.014) and groups concerning employment and money. None of the groups named in the published Table 1 and Fig 4 survive. We have rewritten §3.2 and the abstract around what the corrected analysis supports, and we include a short table reporting the previously named topics with their corrected p and q so that readers can see what changed.

**Manuscript changes.**
- **§2.6.** Add: "p-values are corrected for multiple testing across the 589 tested groups using the Benjamini–Hochberg procedure; we report q-values and treat q < 0.05 as significant." State why 589 of the 1,006 groups are tested.
- **S3.** Record B = 10⁵ and the reason (the Monte-Carlo p-value floor must sit below α/m).
- **Table 1 and Fig 4.** Add a q column / recolour by q (shared with R1.1).
- **§3.2.** Report 157 at p < 0.05, 22 at q < 0.05, 6 at q < 0.01, and that roughly one of the 22 is expected to be a false discovery.

**Figures / tables.** [`tables/R2.6_fdr_counts_draft.md`](tables/R2.6_fdr_counts_draft.md) (+ .csv) — counts at p < 0.05, p < 0.01, q < 0.05, q < 0.01 for Λ and δ². Per-group q-values in [`tables/R1.1_groups_all_draft.csv`](tables/R1.1_groups_all_draft.csv).

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

**2026-10-14 (WS1 → WS4, R1.1 / R2.9).** Step 8 is built. The revised Table 1 and Fig 4 come from two scripts, so curation happens once on the table and the figure follows: `src/8g_group_table.py` → `tables/R1.1_groups_all_draft.csv` (every tested group, every column) plus `tables/R1.1_table1_draft.md` and `tables/R1.1_table1_published_draft.md`; `src/8h_drift_map.py --table ...` → the revised Fig 4, with `--annotate` choosing which groups are named and `--exclude` curating out artifacts (the excluded count is printed on the figure, so a sentence in the text can match it). **WS4: pull topic names, δ, Λ, p and q from `R1.1_table1_draft.csv`, not from the published table.** The all-groups CSV carries two new descriptive columns worth using in the text: `straightness` = δ/Σ‖Δₜ‖ (1 = straight line, ≈1/√(T−1) = random walk, ≈0 = noise around a location) and `max_step`, which flags groups whose movement is a single level shift rather than gradual drift — relevant for how "clinton hillary trump" is described, since a 2016 jump and a steady drift are different claims. **Caveat: `/mnt/sciclone` was unmounted on 2026-10-14, so neither script has been run on real data yet** (both were smoke-tested on a synthetic fixture). Numbers in the drafted R1.1/R2.5/R2.6 responses are the step-7 numbers and should be re-checked against the first real run.

**2026-10-14 (WS1 → WS4, R2.9).** WS1's read on Fig 5: a wording fix is enough, and a test is not worth building. The only sentence that needs one is the abstract's claim that inter-topic change goes "beyond what the null model can explain" — Fig 5 shows no null and no uncertainty, and the Fig 5 quantity (pairwise distances between topic groups over time) is not what the Λ test addresses. Recommend deleting that clause and describing Fig 5 descriptively. If WS4 would rather keep a quantitative claim, WS1 can add a permutation test on the slope of pairwise distance against time (a day's work, same exchangeability null); say so and it goes on the WS1 list.

**2026-10-14 (WS1 → WS4).** Step 8 has been run; two notes for the text. (1) **1002 "clinton hillary trump" is a usable headline example, with the right wording.** It appears in 14 months — 2008-01, 2008-05, eleven consecutive months 2016-02 to 2016-12, then 2020-03 and 2020-11 — and is nearly stationary within 2016 (median 1-month step 0.07) while its cross-cycle steps are 3–5× larger (up to 0.333, more than its net displacement of 0.226); Spearman(gap, step) = 0.86. The alignment step had no access to period, so these centroids were grouped on embedding proximity alone: the same embedding-recognised topic sits in measurably different regions in different election cycles. That is the time-dependence the test detects, and it is a point in favour of the embedding-based approach. Write it as "occupies measurably different regions in different periods"; avoid "drifts steadily" and "gradually", which the test cannot support for any group. 983 and 991 have the same shape. (2) **Name groups by id, not by keyword.** 221 "music listen track" is significant (q=0.003) while the published "music" group 244 is not (q=0.24) — different aligned groups, T=18 vs T=5. "Music drifts" and "music is stable" are both ambiguous now. This is itself an argument the paper can make: three-word TF-IDF labels cannot carry the distinction, which is why the trajectory is measured in the embedding rather than on the labels. The 221/244 contrast would repay a closer look if anyone has time.

**2026-10-19 (WS1 → WS3, R2.8).** `8f_diagnostics.py` now carries a step-size-vs-calendar-gap diagnostic that may be useful for R2.8. Rationale: under diffusion E‖Δ‖² grows with the time separating two observations, under noise around a stable location it does not, so the profile discriminates the two without any extra modelling. Reusable pieces are `gap_steps(periods, steps)`, `gap_rho(gap, length)` and `gap_table(gap, length)` in that script (not in `drift.py` — move them there if WS3 wants them, and note it here). On the published run the profile is flat: median step 0.139 at a 1-month gap, 0.147 beyond 24 months, against a diffusion prediction of 0.848 at the longest gaps. The same machinery diagnosed the two warm-started null runs in R1.1 step 6, where distance *did* grow linearly with lag — so it is a working detector of a drifting sequence, not just a null result. Caveat: measurement noise floors ‖Δ‖, so a flat profile bounds drift rather than excluding it.

**2026-10-19 (WS1 → all).** WS1 is paused at a clean point; see the workstream table for state and the resume commands. `src/drift.py` gained `bh_qvalues`, `path_length`, `straightness`, `max_step`, `load_keywords`, `group_label` since the 2026-09-23 note — all additive, no signature changed, and the locked call in that note is still correct.

**2026-10-19 (WS3 → WS2, R2.1 / R2.2).** R2.1 arm A needs to match topic groups across two different embedding spaces, and R2.2 needs to match them across values of k. Both reduce to the same operation: match groups by the **(month, cluster) members they share**, never by centroid distance (across models the dimensions differ, 384 vs 768). WS2 owns that matcher per the R2.2 plan — WS3 will consume it rather than write a second one. If WS2 would rather not own it, say so here and WS3 will put it in `src/ws3_common.py`. Related: R2.1 arm B2 produces a *directly comparable* group partition over the identical 10,200 (month, cluster) units (same clustering, model-B coordinates, re-aligned), so plain ARI works there with no matching at all — that trick may be useful for R2.2's k=50 replicate too.

**2026-10-19 (WS3 → all, infrastructure).** Three facts found while setting up the cm work.
1. **The cm-side HF cache is `/sciclone/proj-ds/geograd/stmorse/.cache/huggingface`** (inside the pods' `results` mount), and it holds MiniLM. The `HF_HOME` env in `manifests/dev-cpu-create.yml` and `dev-create.yml` points at `/sciclone/home/stmorse/.cache/huggingface`, **which does not exist**; `embed-create.yml` points at a `/hub` subpath of the same non-existent tree. Those runs worked only because `1_embed.py` overrides `HF_HOME` from `config.ini`'s `hf_path`. Anything that relies on the manifest env alone will miss the cache. `config.example.ini`'s `hf_path` is wrong for the same reason — left unedited since it is shared; worth a fix by whoever owns it.
2. `manifests/dev-cpu-create.yml` does `cd /sciclone/geograd/stmorse/dtm`, which is **not mounted in that manifest** (only `home`, `data10` and `proj-ds/geograd/stmorse` are). The reachable checkout there is `/sciclone/home/stmorse/projects/dtm`. WS3's own manifests use that path.
3. **`labels_YYYY-MM.npz` covers every row of every month**, including the 86 months with no metadata (verified against `totals.csv` for 2006-06, 2008-06, 2015-11, 2018-06, 2021-06, 2022-12). So row → cluster is always available offline; metadata is only needed to recover a *comment id*, i.e. for verification. This is what makes R2.1's sampling a cheap local step.
