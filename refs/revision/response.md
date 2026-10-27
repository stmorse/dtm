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
| **WS2** Alignment & k sensitivity | R2.4, R2.2, R1.3 | `src/9_*.py`; SI section and figures | WS1 statistic (can start on partition stability without it) | running: R2.4 baseline + grid + keyword check done 2026-10-26 (`src/9a`–`9c`; results under R2.4; response text not yet drafted). Next: R2.2 (k grid). Matching via WS3's `src/matching.py`; WS2 helpers in `src/ws2_common.py`. |
| **WS3** Scale & model robustness | R2.3, R2.1, R2.8, R1.2 (normalization) | `src/10_*.py`, `manifests/rev-*.yml`; subsample dataset on sciclone; SI sections | WS1 statistic; cm time (start the cm jobs early) | **R2.1 analysis complete 2026-10-26** (arms A, B1, B2; results and tables recorded under R2.1; `src/10c`–`10j`, `src/matching.py` shared with WS2). **R2.8 main matrix complete 2026-10-27** (6 windows × 3 nulls, 11,471 null groups; test calibrated at 4.65%, response and S2 edits drafted; `src/10k_nullrun.py`). R1.2 quick checks done 2026-09-22. **Next: R2.8 n-ladder + R ladder, then R2.3.** R2.1 remainder is write-up plus three small reruns — see the R2.1 open-items list. |
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
- **WCSS / inertia availability across the k grid** (checked 2026-10-26, from `km/km50/models/`; same pattern expected for km30/70/90):
  - `stats_YYYY-MM.pkl` exists for all 204 months, but `avg_inertia` is only meaningful where the fit was full KM. Valid range ~0.76–0.82 for **2006–2013 and 2015**; for **2014 and 2016–2022** it is 0.0000–0.02, i.e. the known MBKM `inertia_`-is-last-batch-only artifact.
  - `wcss_YYYY-MM.json` (exact, from `2c_wcss.py`) exists for **2016–2021 only** (72 months).
  - So a valid WCSS number exists for 2006–2013, 2015 (via `avg_inertia`) and 2016–2021 (via `avg_wcss`), and for **neither source in 2014 and 2022** (24 months). `sandbox2.ipynb` combines the two sources, which is defensible, but the eras come from different code paths.
  - **Implication for Fig S1**, whose caption says "all time periods (excluding 2014-2015)": 2015 *is* available, 2014 genuinely is not, and 2022 has no valid source. WS2 should reproduce the figure and reconcile the caption with what was actually plotted.
- **Monthly volumes:** `save_path/totals.csv` (2006-01 has 2,773 comments).
- **Per-comment metadata (index ↔ comment ID) exists for 2006-01 … 2015-10 only** (checked 2026-09-22):
  - `save_path/metadata/metadata_YYYY-MM.csv`: 2006-01 … 2012-12 (84 months, Apr–May 2025). The files are **gzip-compressed despite the `.csv` name**. Columns: `id, created_utc, parent_id, subreddit, subreddit_id, author, idx`. The script that wrote them is not in the repo.
  - `save_path/metadata2/metadata_YYYY-MM.pkl`: 2007-01 … 2015-10 (106 months, Nov 2024, 68 GB). A list of `[?, author, id, created_utc]`, written by the now-commented-out NPZ path in `1_embed.py`. **Corrected 2026-10-21: the row index is the record's POSITION in the list, not the first field.** The first field is inconsistent across eras — for 2007-06 it equals the position, but for 2015-07 it is the 1M-row batch counter (max 54 for 54.8M rows), matching `1_embed.py`'s `df.append([k, ...])` with `k` the batch count. Position-based lookup is verified against both the `metadata/` CSV (2007-06) and the retrieved text (2015-07), 300/300 each. Also note these files are expensive to read: 1.2–2.4 GB on disk for 2013–2015, ~6x that once unpickled (see the 2026-10-20 OOM note).
  - Row counts match the embeddings and `totals.csv` (checked for 2006-01 and 2007-01).
  - **None for 2015-11 … 2022-12** (86 months, the largest). There, row *i* can be mapped to text only by re-streaming the raw dump with the `1_embed.py` filter.
  - Comment text is in neither file. Retrieving bodies always needs one pass over that month's raw dump. Total dump size is ~1.4 TB compressed; a late month is ~32 GB of `.zst`.
- **Four months of `mbkm_50` were clustered on a chronological prefix of the month** (found by WS3, 2026-10-20): 2020-06 (first 91.9% of comments), 2020-08 (87.9%), **2020-09 (7.4%, i.e. only 2020-09-01 to 09-03)**, 2022-06 (54.3%). The dumps are time-ordered, so a prefix is the month's first days, not a random sample. Labels, centroids and TF-IDF all stop at the same exact 1M-chunk boundary, while the embeddings are complete — the clustering ran before the embedding job had finished writing those zarrs. The `km/km{30,50,70,90}` runs have **complete** labels for all four. Impact and repair options: Cross-workstream notes, 2026-10-20.
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
- [ ] **Four months were clustered on a chronological prefix of the month** (2020-06, 2020-08, 2020-09, 2022-06; worst is 2020-09, whose centroids cover only 2020-09-01 00:00 to 09-03 02:10, i.e. 2.1 days). A bias problem, not a precision one — 12M rows still give a centroid SE of ~0.002. **Decision (user, 2026-10-20): re-estimate those centroid sets from the complete months (option 2), then possibly recluster (option 3); deferred.** Affects 97 of 589 tested groups, 30 of them at a trajectory endpoint; of WS1's 22 significant groups, 7 are touched but only **983** has a prefix month as an endpoint (2020-08, 88% coverage). Found by WS3 2026-10-20 — full detail, impact and repair options in Cross-workstream notes. Needs a decision (disclose vs recluster) and, either way, a sentence in the data description.
- [ ] **2018-03: published `labels` and `model_cc` come from different fits** (4% agreement, found by WS3 2026-10-20). Probably a re-run that rewrote the centroids but not the labels/tfidf, in which case the paper's centroid-based results are fine but 2018-03's TF-IDF **keywords** are stale — which touches group naming. One deferred test settles which file is stale; detail in Cross-workstream notes.
- [ ] S1: WCSS is described as "average distance" but the formula uses squared distances. The 2014–15 exclusion needs an explanation (mixed KM/MBKM, see above).
- [ ] Limitations cites "Appendix 4.2"; it should be S2.
- [x] S2 says "resampled with replacement"; the code permutes and varies the init only (WS3, R2.8). **Resolved 2026-10-27:** corrected S2 wording drafted, and `src/10k_nullrun.py` adds the true with-replacement bootstrap alongside the published init-only null. Also fixed there: S2's "G = k groups" expectation, and the S2 body/S3-caption disagreement about shared vs different initialisation.
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

**[internal]** Workstream: WS3 · Status: **analysis complete (2026-10-26); response text and final figure selection outstanding.** Design agreed with user 2026-10-19; arms A, B1 and B2 all run on all 204 months.

**Open items on R2.1, in priority order.** None blocks another workstream.
1. **Write the response prose and pick the final table set.** Three arm A tables and the arm B tables exist; the response text under *Response* below is still the September draft and does not mention arms B1/B2 or any of the corrections.
2. **Re-run armB1 and armB2 excluding 2018-03 and the four prefix months** (2020-09 and the other three in the 2026-10-20 note). Those months are inside the stratified draw that produced the arm B centroids. Cheap (`--months` already supported, ~5 min) and it removes an unforced caveat. Nothing suggests it matters; we have not checked.
3. **Re-run arm A group keywords at `--kw_max 200`** (`10h --stages groupkw`) before quoting any individual group. Current readings rest on 40 comments per group.
4. **Optional, now only confirmatory:** WS1's `gap_rho` / `gap_steps` per space. The straightness analysis under arm B1 already explains the stronger encoders' extra sensitivity, so this is a second independent check rather than a deciding test.
5. **Optional:** the 31 metadata2 months' comment-ID checks (`10d_retrieve.py --verify_only --workers 1 --metadata_max_gb 4`). The nearest-centroid certificate already covers all 204 months; this is belt-and-braces.
6. **Deferred, not WS3's call:** resampling and re-retrieving 2018-03 to give it a balanced stratified draw, and the prefix-month repair (option 2) from the 2026-10-20 note.

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
- **Manifest:** `manifests/rev-ws3-dev.yml` exists for interactive cm work, but the measured numbers (2026-10-20) make cm unnecessary for R2.1: the full 204-month run is ~4.2 h of retrieval on prospero, bandwidth-bound. A cm Indexed Job stays the fallback if the sshfs mount proves too unreliable for a half-day job (it dropped on 2026-10-14 and again on 2026-10-19), since pods read `data10` over NFS natively. Retrieval is per-month resumable either way: the certificate is written last, and a month with a certificate is skipped.

**Output layout** (all new; nothing under `mbkm_50/` or `km/` is touched). `rev = save_path/rev/ws3`:

| Path | Contents | Size |
|---|---|---|
| `rev/sample/sample_YYYY-MM.npz` | `rows`, `cluster`, `flags` (uniform/stratified bits), params | ~3 MB/mo, ~550 MB |
| `rev/text/text_YYYY-MM.jsonl.gz` | retrieved `{row, id, cluster, body}` | ~26 MB/mo, ~5.3 GB |
| `rev/cert/cert_YYYY-MM.json` | per-month verification certificate | tiny |
| `rev/emb/<tag>/emb_YYYY-MM.npz` | `rows`, `X` (fp16, unit-norm), `meta`; tag ∈ `minilm`, `mpnet`, `bge` | 192 MB/mo (384-d), 384 MB/mo (768-d); ~195 GB total |
| `rev/cc/<tag>/`, `rev/align/<tag>/` | centroids and alignment per arm (not built yet) | small |

`rev/ws3_smoke/` holds the smoke-test copy of the same tree. Local intermediates stay in `output/ws3/` (gitignored); final products go to `refs/revision/{figs,tables}/R2.1_*`.

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

**Arm A results (2026-10-26). Preliminary and diagnostic.** Headline: **month-level topic structure reproduces under all three encoders; individual topic groups do not — and not because of the encoder.** Both findings are only interpretable against a *reproducibility floor*, which is why every table below carries one.

*The floors.* Two MiniBatchKMeans seeds on the identical uniform sample give the cluster-level floor; the same two seeds carried through UMAP+HDBSCAN give the group-level floor. Nothing but a k-means restart separates the two sides.

*Cluster (month) level* — [`tables/R2.1_armA_cluster_level.{csv,md}`](tables/R2.1_armA_cluster_level.md):

| comparison | matched purity | % of floor | ARI | AMI | CKA | kNN overlap | layout ρ | homog. / compl. |
|---|---|---|---|---|---|---|---|---|
| MiniLM seed 0 vs seed 1 (**floor**) | 0.469 | 1.00 | 0.335 | — | — | — | — | — |
| published vs MiniLM-sample (**subset**) | 0.430 | 0.92 | 0.298 | 0.519 | — | — | — | 0.546 / 0.545 |
| minilm vs mpnet (**model**) | 0.382 | **0.82** | 0.228 | 0.431 | 0.684 | 0.368 | 0.707 | 0.440 / 0.441 |
| minilm vs bge (**model**) | 0.321 | **0.68** | 0.182 | 0.372 | 0.615 | 0.283 | 0.587 | 0.381 / 0.381 |
| mpnet vs bge | 0.307 | 0.65 | 0.169 | 0.359 | 0.618 | 0.278 | 0.534 | 0.365 / 0.365 |

*Group level* — [`tables/R2.1_armA_group_level.{csv,md}`](tables/R2.1_armA_group_level.md), compared at the comment level (each sampled comment belongs to one group per space), HDBSCAN noise excluded on both sides:

| comparison | 1-1 purity | % of floor | many-to-one median | ARI | AMI | homog. / compl. |
|---|---|---|---|---|---|---|
| MiniLM seed 0 vs seed 1 (**floor**) | **0.181** | 1.00 | 0.220 | 0.096 | 0.515 | 0.536 / 0.532 |
| minilm vs mpnet | 0.148 | **0.82** | 0.180 | 0.067 | 0.440 | 0.462 / 0.460 |
| minilm vs bge | 0.125 | **0.69** | 0.152 | 0.058 | 0.400 | 0.421 / 0.423 |
| mpnet vs bge | 0.118 | 0.65 | 0.147 | 0.045 | 0.391 | 0.412 / 0.415 |

**The ratio is the result.** Absolute agreement falls with granularity (0.47 → 0.18) exactly as expected, but the *relative* cost of changing encoder is constant at both levels: mpnet retains 82% of the floor, bge 68–69%. The group level fails to reproduce *within one encoder* — median 22% self-overlap under a k-means restart, only 9.4% of groups surviving at ≥50% — so group non-correspondence is a property of the UMAP+HDBSCAN step, not of the encoders. Arm B2 reaches a **weaker** conclusion by a different route: over the identical 10,200 (month, cluster) units, grouping agreement on units that are non-noise in both is ARI 0.52 for a different encoder against 0.71 for the alignment seed alone — substantial, not near-zero, and the same 73%/67%-of-ceiling ratio. The arm A floor measured here is at the **comment** level and compounds cluster instability with grouping instability, so the two are consistent rather than contradictory: both steps contribute, and the alignment step contributes more. See the arm B1/B2 section below.

*Group structure and drift* — [`tables/R2.1_armA_group_structure.{csv,md}`](tables/R2.1_armA_group_structure.md):

| space | groups | noise | median span | tested | median T | median δ | q<0.05 | rate | corresponds from minilm | chance | enrichment | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minilm | 1009 | 0.251 | 105.5 mo | 630 | 8 | 0.153 | 77 | 12.2% | — | — | — | — |
| mpnet | 1025 | 0.251 | 100.5 mo | 644 | 8 | 0.165 | 53 | 8.2% | 22/77 = 28.6% | 8.2% | **3.47×** | 2e-07 |
| bge | 997 | 0.252 | 102.5 mo | 618 | 8 | 0.124 | 93 | 15.1% | 24/77 = 31.2% | 15.1% | **2.07×** | 3e-04 |

Aggregate structure is near-identical across spaces (group count, noise fraction, span and T distributions). Only the significance *rate* is comparable, since each space derives its own groups, and all three fall in 8–15%. Group-level significance transports **above chance but not reliably**: ~29–31% of MiniLM's significant groups have a best-overlap counterpart that is also significant, a 2.1–3.5× enrichment. Exactly what a 0.18 purity floor predicts — group-level significance cannot transport better than group identity does.

*Qualitative check (the most persuasive evidence, and keyword-free in its matching).* Matching clusters one-to-one by **member overlap** and reading TF-IDF keywords only as labels, matched clusters are recognisably the same topics even at ARI 0.18–0.26. 2016-11: `deck dps ult decks buff` ↔ `deck dps ult mana crit` (71%), `obama botsbyliam donald clinton` ↔ `clinton hillary obama donald election` (70%), `que não det du les` ↔ `que det du der les` (92%). 2020-04: `trump biden bernie president election` is the identical top-5 in both spaces. So ARI understates topical correspondence substantially — the disagreement is about which comments land where, not about which topics exist.

*Merges vs splits are symmetric.* Homogeneity and completeness are equal to three decimals in every comparison at both levels, so the disagreement is symmetric boundary noise, not one encoder systematically merging what another splits. Individual counter-examples exist (mpnet absorbs MiniLM's Westworld cluster; neither mpnet nor bge isolates MiniLM's HTML-markup cluster) but they are local.

*Era effect, and a link to R2.3.* The subset effect is strongly size-dependent — published vs MiniLM-sample purity runs 0.345 (2006–08) to 0.479 (2020–22), Spearman +0.784 with month size — while cross-encoder agreement is flat. For 2006–07 the uniform draw *is* the whole month, so this is not sampling: **the published early-month partitions are poorly determined in their own right** (k=50 on a few thousand comments). That is an R2.3 finding arriving through R2.1.

*What the drifting groups are about.* The arm A groups with the strongest time-dependence are dominated by **platform, media-format and site-meta** themes rather than topical discourse: Digg→Reddit migration (`digg reddit redditors comment programming`, T=25, δ=0.329), browser technology (`browser firefox javascript`, δ=0.278), Reddit site features (`reddit page link bestof iama`), moderation templates, YouTube video and gif sharing, karma/repost meta, plus a meme-era group (`poker megusta buzz troll`, datable to ~2010–12). This **independently reproduces WS1's observation** that most published groups surviving BH were platform/format/language groups — now from re-derived groups in a different embedding space. Among the groups that both reproduce and show time-dependence, covid (`covid vaccinated coronavirus`, 0.76 overlap minilm↔bge, significant in both) is the clearest substantive topic; the rest are bot templates, non-English language groups and compliment formulas.

*Measures used, with failure modes (all matching is by shared comments; keywords are labels only).*
- **Matched purity** (Hungarian 1-1): readable as a proportion, but forces 1-1 so a genuine merge is penalised arbitrarily, and it cannot fall to chance.
- **Many-to-one overlap**: each group's best counterpart — an upper bound, since many groups may map to the same counterpart.
- **ARI**: harsh with many clusters (at k=50, and especially at ~1000 groups, almost all pairs are "apart"), and not a proportion of anything. **It understates topical correspondence badly here** — see the keyword tables.
- **AMI**: tolerant of refinement, so high AMI can mean agreement *or* that one space subdivides the other; cannot distinguish those alone.
- **Homogeneity / completeness**: the asymmetric pair, which is what separates merges from splits.
- **Jaccard on significant sets: avoid.** It is confounded with set size — with |A|=18 and |B|=115 it cannot exceed 0.157 even under perfect nesting. Containment is the right measure *if* the sets have to be compared at all. **Better: don't compare BH-significant sets across spaces.** At 589 tests the q<0.05 set is a knife-edge function of the whole p-distribution, so its membership moves on permutation noise (see arm B2: mpnet has 24 at q<0.05 against minilm's 53 while holding *more* groups at both p<0.05 and p<0.01). Compare **rate at a fixed p**, and rank-correlate δ/Λ/p over all tested groups.
- **CKA** and **kNN overlap**: encoder-level similarity with no clustering at all, so no circularity. CKA is a global second-order summary and can be high while local neighbourhoods differ; kNN overlap depends on k and is deflated by near-ties.

*Caveats.* The keyword view rests on 40 comments per group (reservoir-sampled across months) against the published pipeline's 100 nearest-to-centroid comments, so individual group readings should be re-checked at a larger `--kw_max` before quoting. Arm A uses MBKM on a 200k sample where the published run used `partial_fit` over the full month. UMAP is seeded here (which forces single-threaded) where the published alignment was not. An automated attempt to classify groups as interpretable/artefact by word lists was uninformative (73 of 77 fell to "other"); the platform/format pattern above is a reading of examples, not a quantified claim — hand-coding the ~77 groups is the honest route if we want a number. Group-correspondence counts shift by ±1–2 with the pooled comment subsample (bge 23 or 24 of 77 across draws).

*Reproducing all of it* (run from `src/`, in order; each is resumable):
```
10c_sample.py --seed 0                          # sample index sets
10d_retrieve.py --seed 0 --workers 8            # text (certificates verify alignment)
10e_embed.py --seed 0                           # three encoders, GPU
10f_centroids.py --seed 0                       # arm A k-means (2 seeds) + arm B centroids
10g_compare.py --seed 0 --stages armA           # arm A alignment + drift per space
10h_arma_inspect.py --seed 0 --stages month,rsa,groups,grouplevel,topics,groupkw
10i_armA_summary.py --seed 0                    # the three summary tables above
```
`10h --stages topics` takes `--inspect_months`; `groupkw` takes `--kw_max`. Supporting drafts: `R2.1_armA_month_agreement{,_byyear}_draft.csv`, `R2.1_armA_rsa_draft.csv`, `R2.1_armA_groups_<space>_draft.csv`, `R2.1_armA_group_keywords_draft.csv`, `R2.1_armA_topics_<month>_draft.json`.

**Arm B1 and B2 results (2026-10-26). Preliminary and diagnostic.** Headline: **the drift finding survives every change we can make to the encoder and to the grouping.** Arm B1 shows the paper's own topics keep their measured movement; arm B2 shows that re-grouping the same clusters into a largely different set of groups recovers the same drift rate. Two reporting errors found along the way are recorded below, because both would have misled the response.

*`published` in these tables means the published centroids and published group labels run through **WS1's new locked statistic** — not the paper's reported p-values.* With B = 10⁵ it gives 155 at p<0.05, 58 at p<0.01, 19 at q<0.05, 6 at q<0.01, against WS1's locked 157 / 58 / 22 / 6. p<0.01 and q<0.01 match exactly; the p<0.05 and q<0.05 gaps (2 and 3) are Monte Carlo noise between two correct implementations with different per-group RNG streams. **That is itself a calibration worth citing: even at B = 10⁵, two correct runs disagree on q<0.05 membership by ~15%.**

**Use B = 10⁵, not 10⁴.** Our first pass ran `--n_perm 10000` and reported q<0.05 counts from it. That repeats the failure WS1 already documented under R1.1: with 589 tests BH's tightest threshold is 0.05/589 = 8.5×10⁻⁵, below the smallest attainable p of 1.0×10⁻⁴. At 10⁴ the published column's entire significant set sits between p = 0.0001 and 0.0015 — a window **15 permutation counts wide** — so q-set membership turns on permutation noise. `--n_perm` now defaults to 100000 in `10g_compare.py` with that reasoning in the help string.

**Arm B1: the published groups, re-coordinated.** 589 tested groups, identical in every column by construction; `minilm` is the sampling-noise control (arm B centroids re-estimated from the stratified draw in MiniLM space).

| | published | minilm (sampling control) | mpnet | bge |
|---|---|---|---|---|
| median δ | 0.1831 | 0.1655 | 0.1566 | 0.1165 |
| median Λ | 0.5409 | 0.5181 | 0.6068 | 0.6765 |
| p < 0.05 | 155 | 161 | 191 | 224 |
| p < 0.01 | 58 | 68 | 85 | 116 |
| q < 0.05 | 19 | 29 | 66 | 116 |

- **δ ranks are highly stable, and the model effect is smaller than the sampling effect.** Spearman on δ over all 589: published↔minilm **0.956**, minilm↔mpnet **0.989**, minilm↔bge **0.980**, mpnet↔bge **0.983**. Λ 0.78–0.94, p 0.73–0.93.
- **Absolute δ is NOT comparable across encoders.** bge compresses every length scale by the same factor — median ratio to published: δ 0.677, path length 0.640, max step 0.657, diameter 0.670. Quoting bge's median δ of 0.117 against the published 0.183 as "less drift" would be wrong. **Report scale-free quantities across spaces.**
- **Shape is nearly invariant.** Straightness (δ ÷ path length) median: published 0.1721, minilm 0.1728, mpnet 0.1786, bge 0.1855.
- **The extra sensitivity of the stronger encoders has an identifiable, non-artefactual cause.** The straightness gain is small, systematic, and absent from the pure-sampling control: paired Wilcoxon vs published gives mpnet **+0.0046** (60% of groups higher, p=4e-08) and bge **+0.0088** (66%, p=8e-19), against minilm +0.0008 (52%, p=0.27). It is that gain which drives the extra significance, not a lower noise floor: Spearman(Δstraightness, ΔΛ) = +0.30 to +0.38, groups newly crossing p<0.05 have 2–6× the straightness gain of the rest, and the noise-to-signal ratio is flat (centroid SE ÷ median max step = 0.100 / 0.111 / 0.113 for minilm / mpnet / bge). More significance in a better space is evidence the signal is real; the worry to rule out was a noise reduction inflating the statistic, and it is ruled out. **This largely answers the open gap-profile question** — WS1's `gap_rho` per space would be a useful independent check but is no longer load-bearing.
- **Named groups hold everywhere.** All six (221, 821, 856, 983, 991, 1002) stay p ≤ 0.009 in all four spaces. 221/856/991 are q<0.05 in all four; 983/821 in three; 1002 in two. Where they drop out it is BH knife-edge (1002: q=0.0713 at p=0.0069).

**Correction to the containment claim recorded earlier today.** The 2026-10-26 note said the significant sets are "nested at 94–100% containment, published ⊂ minilm ⊂ mpnet ⊂ bge". The chain among the three *re-estimated* spaces holds — minilm ⊂ mpnet ⊂ bge at ~94% — and all 18 published-significant groups (at B=10⁴) are inside bge's set. But **published ⊂ minilm is only 56%**, so there is no clean four-way chain. The shortfall is BH multiplicity under sampling noise, not disagreement: the 8 missing groups have minilm p of 0.003–0.016 against published p of 0.0002–0.0015, 26 of the 31 q-flips are p<0.05 on both sides, and the degradation is symmetric (published→minilm median p inflation 1.39×, 58% worse; minilm→published 1.68×, 66% worse, with minilm holding *more* groups at p<0.01, 65 vs 57). **Do not report q-set containment at all** — see the next item.

**BH q-counts are not a usable statistic for comparing groupings or spaces.** In arm B2 the counts run 19 / 30 / 53 / **24** / 89, with mpnet *below* minilm despite having more groups at p<0.05 (206 vs 186) and at p<0.01 (79 vs 75). At rank 50 mpnet's p is 0.00581 against a required 0.00409 while minilm's is 0.00364 against 0.00394 — opposite sides of one knife edge, worth a 29-group swing. **Report rate at a fixed p**, and reserve BH for choosing which groups to name inside a single analysis.

**Arm B2: re-grouping the same 10,200 units.** Each row's drift is measured on its own re-derived grouping in its own coordinates. `published_realigned` is the control: same coordinates, same units, only the UMAP/HDBSCAN seed and code path differ.

| grouping | what changed | groups | noise | tested | median T | median δ | **rate p<0.05** | rate p<0.01 |
|---|---|---|---|---|---|---|---|---|
| published | — | 1006 | 0.279 | 589 | 8 | 0.1831 | **26.3%** | 9.9% |
| published re-aligned | alignment seed only | 1022 | 0.289 | 621 | 8 | 0.1798 | **26.1%** | 9.3% |
| minilm | + centroid sampling | 1034 | 0.283 | 635 | 8 | 0.1639 | **29.3%** | 11.8% |
| mpnet | + encoder | 1018 | 0.263 | 611 | 8 | 0.1529 | **33.7%** | 12.9% |
| bge | + encoder | 1025 | 0.248 | 630 | 8 | 0.1150 | **33.5%** | 15.9% |

**This is the strongest robustness result in R2.1, and it conditions on nothing MiniLM chose.** The grouping is largely re-carved — only about half the pairwise structure is preserved, see below — yet the fraction of groups with significant time-dependence stays in a 26–34% band at every threshold, with median T, median δ (after the bge scale compression), group count and noise fraction stable to a few percent. The drift finding is a property of the corpus, not of the encoder and not of our alignment choices. The monotone rise with encoder strength is the same straightness effect as in B1, now on independently re-derived groups.

**Correction: our first armB2 grouping numbers understated the agreement badly.** We reported ARI 0.29 for the seed-only control and 0.11–0.13 for the three spaces, and concluded "the grouping is almost entirely replaced". **That was an artefact of the statistic.** HDBSCAN marks 25–29% of units as noise, and sklearn's ARI treats `-1` as one cluster of ~2,800 units, so disagreement about *which* units are groupable swamps the grouping comparison. Restricted to units non-noise in both:

| vs published | n core | ARI (all units) | **ARI (core)** | AMI (core) | homogeneity / completeness | noise-set Jaccard |
|---|---|---|---|---|---|---|
| re-aligned (seed only) | 6,106 | 0.290 | **0.710** | 0.851 | 0.957 / 0.950 | 0.416 |
| minilm | 5,725 | 0.126 | **0.521** | 0.709 | 0.918 / 0.905 | 0.282 |
| mpnet | 5,863 | 0.124 | **0.519** | 0.707 | 0.908 / 0.906 | 0.274 |
| bge | 5,926 | 0.110 | **0.477** | 0.682 | 0.903 / 0.898 | 0.259 |

Chance baselines are ≈0.000 for ARI and AMI (size-preserving shuffle); homogeneity and completeness are **not** chance-adjusted and run ~0.69 under the same shuffle at this group count, so they are only readable as the excess (+0.21 to +0.27) and are reported with their null in the table. The encoder cost reads the same as everywhere else: mpnet retains **73%** of the seed-only ceiling, bge **67%** — within a few points of arm A's 82% / 68%. The disagreement is **symmetric boundary reshuffling, not refinement**: homogeneity ≈ completeness in every row, each published group spreads over ~1.9–2.0 re-derived groups and vice versa, and the group size distributions are indistinguishable (median 5–6, mean 7.1–7.5). What is genuinely unstable is **which units are groupable at all** (noise Jaccard 0.26–0.42) — a real finding about the alignment step, but a different claim from "the groups do not reproduce".

`grouping_scores()` in `10g_compare.py` now always returns both the all-units and the core figures with the homogeneity null attached, so this cannot be reported one-sided again. **This applies to R2.4 and R2.2 as well** — any ARI over HDBSCAN output with noise included is measuring mostly the noise boundary.

**Arm B2 qualitative check: the groupings agree on content far more than on membership.** Because B2's units are identical across spaces, the **published** TF-IDF keyword list (20 terms per unit) applies unchanged to every grouping — one fixed vocabulary, nothing re-estimated. Jaccard is safe here and only here, since every profile is exactly 20 terms.

- **Every grouping is equally keyword-coherent.** Median within-group mean pairwise Jaccard, against a size-preserving shuffle null: published 0.1245 (75.1×), re-aligned 0.1253 (75.3×), minilm 0.1244 (72.8×), mpnet 0.1247 (74.0×), bge 0.1185 (67.7×), with 99% of groups above null in every case. Identical to three decimals. The re-derived groupings are **equally coherent carvings, not degraded ones** — which rules out the reading that one of them is not tracking topic.
- **Matched pairs share content well beyond their membership overlap.** Hungarian-matched on shared units: published→minilm containment 0.469 / keyword overlap 0.429; →mpnet 0.444 / 0.379; →bge 0.423 / 0.379; →re-aligned 0.735 / 0.600. The mismatched-pair null has **median 0** — over half of random pairs share no terms at all out of 20. 26–28% of matched pairs share under half their units while clearly describing the same content.
- **Tracing named groups is the convincing form** (`R2.1_armB2_fate_draft.csv`). 221 "music listen track song youtube" fragments into 6–8 groups in every space — including 4 under the seed-only control — but every destination is a music group (`youtube music song listen`, `music listen songs album`, `music listen track song`): pure granularity, driven by the alignment seed, not the encoder. 856 "account money customer order email" → minilm puts 13 of 14 members in one group (`account customer money scam`). 1002 "clinton hillary trump campaign bernie" → every space recovers a Trump/Clinton/Bernie group *and* separately splits off an Obama/McCain group (`obama mccain campaign barack` in all three), consistently separating 2008-era from 2016-era US politics where the published grouping merged them. 991 "english europe language european canada" → all three spaces split it the same way, into language/accents and European geography. 821 → wages/jobs vs rich/wealth; 983 → interview/political vs president/criminal.
- **So the low membership overlap is boundary placement and granularity over coherent content**, and several of the splits are substantively interpretable — arguably finer-grained than the published grouping.

*Caveats specific to arm B.* The keyword profiles are the **published** TF-IDF, from MiniLM's 100 nearest-to-centroid comments per unit. That is deliberate — one fixed vocabulary is what makes "is the disagreement semantic?" answerable — but it is not independent of MiniLM, and cannot detect a case where mpnet grouped units that MiniLM's keywords describe identically. Arm B centroids are sample means of the stratified draw under the nearest-published-centroid assignment, which is a different estimator from the published MBKM centroids (not only noisier). The 2018-03 defect and the four prefix months are still inside the stratified draw; excluding those 5 months and re-running armB1 is cheap and not yet done. UMAP is seeded here (forcing single-threaded) where the published alignment was not.

*Reproducing arm B* (run from `src/`, after `10f_centroids.py`):
```
10g_compare.py --seed 0 --stages armB1,armB2 --n_perm 100000 --keywords
10j_armB_inspect.py --seed 0 --match_top 1200
```
Tables: `R2.1_armB1_{groups,agreement}_draft.csv`, `R2.1_armB2_grouping_draft.csv`, `R2.1_armB2_drift_<grouping>_draft.csv` (5 files), `R2.1_armB2_coherence_draft.csv`, `R2.1_armB2_match_summary_draft.csv`, `R2.1_armB2_match_<a>_<b>_draft.csv`, `R2.1_armB2_fate_draft.csv`. Grouping labels are cached at `rev/ws3/align/armB2_labels_{control,minilm,mpnet,bge}.npz`.

> **Gotcha, cost us a silent wrong result.** `stage_armB2` writes grouping labels to `rev/ws3/align/`, which `--out_path` does not govern — so a 48-month smoke run overwrote the production 10,200-unit labels with 2,400-unit ones, and the keyword analysis then ran on 2006–2009 only while looking entirely plausible. Both sides are now guarded: an incomplete month set writes `smoke<N>_armB2_labels_*.npz`, and `10j` raises if a labels array's length does not match the published grouping. **Any WS script writing under `save_path/rev/` must tie its output path to the month set, not just to `--out_path`.**

*Analysis stages built and smoke-tested (2026-10-21).* `10f_centroids.py` (per-month artifacts) and `10g_compare.py` (all comparisons and tables). Both run clean on the smoke tree; `10g` has selectable `--stages agree,armB1,armB2,armA,suff`, each writing its own table to `refs/revision/tables/R2.1_*`.
- **Membership comes from the nearest published centroid, not the `labels` file** (`10f` stage 0, computed once per month from the MiniLM embeddings and reused by every space). This keeps every month consistent with the centroids the published alignment actually used, and makes 2018-03 correct with no special case.
- **`10f` outputs per (model, month):** arm B centroids from the stratified draw, plus two disjoint half-samples and the `--ladder` subsample rungs for the sufficiency argument; arm A `MiniBatchKMeans(k=50)` on the uniform draw for `--n_seeds` seeds, with the partition of the sampled rows saved so agreement can be computed later. `n_init=1, batch_size=10000` — `n_init=1` matches the published run's `partial_fit` (a single init) and is the fastest setting at production size (4.3 s per fit for 200k x 768). Cost: ~15 s per (model, month), so **~2.5 h for 204 months x 3 spaces**.
- **Ladder rungs must sit below the stratified draw size.** A rung at or above it takes the whole sample, so its distance to the full centroid is exactly 0 — the full draw is the reference, not a rung. Default is now 250/500/1000 against a 2,000 draw.
- **The `agree` stage reports three quantities that must be read together:** published vs MiniLM-sample (the subset effect, i.e. the reviewer's sufficiency question), seed i vs seed j within one space (the within-model instability floor), and the cross-model pairs (the model effect). Everything is by membership (ARI/AMI); centroid vectors are never compared across spaces, which have different dimension.
- **`armB2` includes a control that the smoke test proved necessary:** re-aligning the *published* centroids themselves with a seed. On the 2-month smoke subset the control scored ARI 0.0185 against the published labels — the same as the model-B spaces — showing the agreement floor there is set by the truncated subset and by UMAP seeding, not by the encoder. Without the control, a low cross-model ARI would be misread as a model effect. (The smoke numbers are not results: 2 months slice 100 units out of 10,200 and fragment every group, which is why `10g` prints a warning when given a subset.)
- `10g` also takes a month subset deliberately, so the stages stay testable; it warns that trajectories are truncated and the output is a smoke test rather than the analysis.

*Build and smoke test (2026-10-20).* Stages: `10c_sample.py` → `10d_retrieve.py` → `10e_embed.py`, with `10z_check_filter.py` (filter equivalence), `10z_smoke.sh` + `10z_smoke_check.py` (end-to-end). **Smoke test passes: 0 failures.** Months 2006-06 (zarr, metadata, whole month sampled), 2008-06 (npz, metadata, stratified draw binds), 2015-11 (no metadata, truncated via `--limit_rows`), × 3 encoders, written to `rev/ws3_smoke/` so nothing collides with the production run.
- **Verification is exact.** Nearest-centroid agreement **1.0000** on all three months (chance 0.02), id check 600/600, and re-embedded vs stored MiniLM cosine **1.00000** (median and min). The fp16 boundary flips we budgeted for did not materialise.
- **The certificate needs no embedding reads.** MBKM labels *are* nearest-centroid assignments, so re-embedding a few hundred retrieved bodies and assigning each to its nearest published centroid for that month confirms the row alignment; misalignment would collapse agreement to chance. This covers the 86 months with no metadata, which was the open gap in the verification plan.
- **Scan rate, measured per format** (fast path, one worker): bz2 52–69k rows/s, xz 86k, zst 116k. Weighted over the corpus (12.77 B rows, 1.50 TB: 50 zst months hold 8.38 B rows, 143 bz2 months 3.37 B, 11 xz months 1.01 B) that is **40 core-hours**, and the sshfs ceiling (~100 MB/s) caps the wall clock at **~4.2 h** for all 204 months at 10+ workers — more workers do not help. Embedding adds ~4.2 h on the 5090 (MiniLM 21k/s, mpnet 6.2k/s, bge 8.0k/s measured on real text at max_seq 256). **Full run ≈ 8–9 h, so all 204 months is comfortably sub-day** and the pilot-then-extend split is unnecessary.
- **Implementation note, not a pipeline bug.** `line, buf = buf.split(b'\n', 1)` in a loop copies the remaining buffer per line, so it is quadratic in the read-block size: raising the block from 8 KiB to 4 MiB for sshfs took line iteration from 94k to 13k lines/s. WS3's reader splits each block once and carries the tail. `1_embed.py` and `utils.read_sentences` use the same pattern but at 8 KiB, where the copy is small — they are slow over sshfs, not wrong, and nothing in the published results is affected.
- **Finding for R1.2 / §2.1: the filter keeps *empty* bodies.** `'body' in entry` is true for `body == ''`, so empty comments pass and, like `[removed]`, all embed to a single point. Rare — 2 of 20,430 sampled rows in 2006-06 (~1e-4), 0 in the 2008-06 and 2015-11 samples. Same artifact family as the step-1 finding; the §2.1 sentence should say "non-empty" only if we change the filter, which we should not. Worth one clause in the filter description.

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
*Decisions so far (user, 2026-10-26; revisit when R2.2 starts).* **Use the 192 months common to every k.** km90 has no 2014 at all (its 192 months are 2006–2013 and 2015–2022), and per `km/readme.txt` 2014–15 was a mix of KM and MBKM across the grid. So all k, plus a re-aligned mbkm_50 as the k=50 reference, would be aligned on the same 192 months, with 2014 dropped everywhere. Match **through comments** (a seeded row subsample of each run's `labels`), not centroids: a k=30 cluster is roughly a union of k=50 clusters, and nearest-centroid matching throws away how the comments split. Run the k=50 replicate (km50 vs mbkm_50) first as the floor. Exclude the four prefix months and 2018-03 from anything that uses mbkm_50 labels. Spot-check km labels against their own centroids first; only mbkm_50 has been audited. The existing `km50/align` and `km70/align` are unseeded (Sep 2025), so use them only as a cross-check.  
*Audit (2026-10-27; `output/ws2/audit_kgrid.csv`; centroid and label shapes read from the npz headers).*
- **km30/50/70 are complete.** All 204 months, centroid count = k, and label count = `totals.csv` in every month, so they have none of mbkm_50's prefix problem. TF-IDF is present for every month.
- **km90 lacks exactly 2014 (12 months), and was simply never run for it.** No `cluster_k90_2014` log exists, while 2006–13, 2015 and 2016–22 each have one. Everything else matches as above. **Fill:** `manifests/rev-ws2-km90-2014.yml` runs the unchanged `2c_cluster.py --model mbkm --n-clusters 90` (the same call used for km30/50/70's 2014) on cm, writing to `save_path/rev/ws2/km90/`, not `km/`. With it, all four k cover 204 months.
- **Clustering method by era, from `avg_inertia`, corrects `km/readme.txt`:** 2006–2013 full KM for every k; **2014 MBKM for every k** (not mixed); **2015 KM for km50 only**, MBKM for km30/70/90; 2016–2022 MBKM for every k. So across the k grid the single inconsistent year is **2015 in km50**. For the k=50 replicate, note that mbkm_50 itself is KM for 2007–2014 (`2_cluster.py`) and MBKM for 2006 and 2015–2022, so the replicate compares like with like only in 2007–2013 and 2016–2022.
- **The 192-month decision can be revisited once the fill lands:** with km90's 2014 in place, 204 months are available for every k, and 2014 is the same method (MBKM) at every k.

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

**[internal]** Workstream: WS2 · Status: drafted  
*Design (agreed 2026-10-26).* Vary **one parameter at a time** around the published setting, with ~10 seeds each. Aim for efficiency, not exhaustiveness. Add `max_cluster_size` {20, 50, 100, none} and `min_samples` to the grid: `max_cluster_size` is the parameter the authors actually varied (`max_20` vs `max_100`). Read every parameter effect against the **seed-only spread** at the published setting, since the published grouping is itself one unseeded draw. All agreement measures come from WS3's `src/matching.py`. `pair_scores` (noise dropped pairwise, with the `rng` null) is always reported next to `noise_summary`. Drift stability: per-variant rate at p<0.05 and p<0.01, plus Spearman on δ/Λ/p over all groups matched to published ones by shared units (`match_many_to_one` on raw labels, so `frac_to_noise` is real). BH-significant sets are not compared as sets. The keyword check uses the member clusters' published TF-IDF as labels only. No-UMAP baselines: HDBSCAN on cosine distance, and on PCA-50.

*Baseline (2026-10-26)* — `src/9a_baseline.py --seed 0 --drift` → `output/ws2/9a/baseline.json`, [`tables/R2.4_baseline_draft.csv`](tables/R2.4_baseline_draft.csv). 58 s in total.
- **Published grouping:** 1,006 groups (`np.amax` = 1005, the known off-by-one); noise 0.279; group size median 5, p90 13, max 83, so `max_cluster_size=100` never binds; 589 tested (T ≥ 5), covering 58% of units; median T 8, median span 106 months; 7% of tested groups have more than one unit in some month.
- **HDBSCAN on the stored UMAP embedding (`cu.npz`) does not reproduce the published labels exactly, but nearly does:** ARI 0.990, purity 0.990, noise Jaccard 0.956, with 998 vs 993 groups on units both place. Most of the 6,770 differing label values are renumbering. Probably an sklearn version or tie-breaking difference from the cm environment. So ~99% of the published partition is pinned down by the UMAP embedding, and **essentially all seed sensitivity is UMAP's**. It also means published group IDs cannot be regenerated bit-for-bit, even from `cu.npz`.
- **Battery sanity checks.** Self vs self gives 1.0 on everything. Against a size-preserving shuffle of the group labels: ARI 0.0001, AMI 0.0003, **1-1 purity 0.143, many-to-one median 0.20, homogeneity/completeness 0.661**. These are the chance levels for this grouping. Many-to-one median = 0.20 under the null simply reflects the median group size of 5, and it reads 1.0 whenever most groups survive intact, so it is coarse here; 1-1 purity and ARI/AMI on non-noise units are the informative ones.
- **Seeded re-alignment at the published settings:** deterministic (two runs identical) and **identical to WS3's arm B2 control** (same seed and settings, different code path). 11 s per alignment. Against published, on units both place: **ARI 0.710, AMI 0.851, 1-1 purity 0.763, homogeneity 0.957 / completeness 0.950 (null 0.688 / 0.683)**. Noise: 0.279 vs 0.289, noise Jaccard 0.417, so 40% of units are noise in at least one of the two groupings. **Which units get grouped at all is the least stable part, as WS3 found.** 1,022 groups, 621 tested.
- **Drift wrapper** (`ws2_common.drift_table`, WS1's locked call, B = 10⁵, per-group RNG stream `[seed, coi]` as in WS3's 10g): T, δ and Λ **identical** to WS1's locked run for all 589 groups; p Spearman 0.9999, max |Δp| 0.007. Counts are 155 / 58 / 19 at p<0.05 / p<0.01 / q<0.05, vs WS1's 157 / 58 / 22: the same Monte Carlo gap WS3 reported. 38 s per grouping, so **B = 10⁵ is affordable for every variant**. The full one-at-a-time grid (~120 alignments + drift) is ~2 h.

*Grid results (2026-10-26).* `src/9b_align_grid.py --seed 0 --n_seeds 10` (14 min, 152 groupings, drift at B = 10⁵ for each) → [`tables/R2.4_grid_summary_draft.md`](tables/R2.4_grid_summary_draft.md) (median [min, max] over 10 UMAP seeds), per-run detail in `tables/R2.4_grid_runs_draft.csv`, labels/drift/matches in `output/ws2/9b/`. Keyword check: `src/9c_keyword_check.py` → `tables/R2.4_keyword_{summary,examples}_draft.csv`. Agreement is on units non-noise in both groupings; "rate" = share of tested groups at p<0.05 (published: 155/589 = **0.263**).

| variant | groups | noise | ARI vs pub | noise Jacc | drift rate p<.05 | ρ δ | ρ Λ |
|---|---|---|---|---|---|---|---|
| **seed only** (published settings) | 1019 [960, 1035] | 0.286 | 0.686 [0.666, 0.730] | 0.43 | **0.262** [0.249, 0.293] | 0.70 | 0.41 |
| n_neighbors 10 / 30 / 50 | 1061 / 958 / 897 | 0.26 / 0.32 / 0.34 | 0.66 / 0.64 / 0.61 | 0.35–0.38 | 0.257 / 0.260 / 0.256 | 0.57–0.69 | 0.29–0.38 |
| min_dist 0 / 0.25 | 1035 / 977 | 0.26 / 0.32 | 0.66 / 0.69 | 0.37 / 0.43 | 0.256 / 0.266 | 0.69 / 0.70 | 0.37 / 0.39 |
| align_dim 5 / 20 | 1020 / 1020 | 0.28 / 0.29 | 0.69 / 0.69 | 0.43 / 0.42 | 0.263 / 0.260 | 0.70 / 0.71 | 0.41 / 0.42 |
| max_cluster_size 20 / 50 / none | 1072 / 1025 / 995 | 0.33 / 0.29 / 0.28 | 0.66 / 0.68 / 0.60 | 0.42 | 0.253 / 0.258 / 0.255 | 0.70–0.73 | 0.40–0.41 |
| min_cluster_size 5 / 10 | 416 / 187 | 0.34 / 0.43 | 0.64 / 0.53 | 0.41 / 0.34 | 0.335 / 0.423 | 0.58 / 0.45 | 0.27 / 0.13 |
| min_samples 1 / 5 | 1344 / 491 | 0.23 / 0.35 | 0.62 / 0.65 | 0.37 / 0.42 | 0.228 / 0.328 | 0.74 / 0.59 | 0.43 / 0.30 |
| no UMAP: HDBSCAN cosine / PCA-50 | 352 / 377 | **0.69 / 0.65** | 0.47 / 0.45 | 0.31 | 0.313 / 0.310 | 0.20 / 0.33 | −0.03 / 0.05 |

Reading it:
1. **The UMAP parameters and the seed are the same size of effect.** Over n_neighbors 10–50, min_dist 0–0.25 and align_dim 5–20, agreement with the published grouping (ARI 0.61–0.69) stays within or just below the seed-only band (0.67–0.73, seed-vs-seed 0.72). n_neighbors acts monotonically (more neighbours → fewer, larger groups, more noise). align_dim does nothing measurable. `max_cluster_size` 20 or 50 barely changes anything (ARI 0.93 / 0.99 against the same-seed published-setting run); with no cap there is an occasional large merge (one seed gives 896 groups).
2. **HDBSCAN granularity is the parameter that matters, and it changes the question rather than the answer.** Raising min_cluster_size / min_samples gives fewer, larger groups (187–491). The drift rate rises to 0.33–0.42 because longer trajectories give the test more power. min_samples = 1 gives more, smaller groups and a slightly lower rate (0.23). This is a resolution choice, and it should be described as such.
3. **The aggregate drift result is invariant.** At every UMAP setting and every `max_cluster_size`, the rate at p<0.05 is **0.25–0.27**, matching published (0.263); at p<0.01 it is 0.09–0.11 (published 0.098). This matches WS3's arm B2 (26–34%).
4. **UMAP earns its place.** Without it, HDBSCAN leaves **65–69% of centroids as noise** (vs 28%), and the groups it does find track the published ones poorly (ρ_Λ ≈ 0).
5. **Per-group results are less stable than the aggregate, and the reason is membership, not the statistic.** Λ is exactly reproducible for identical groups (max |ΔΛ| 3×10⁻¹⁵ over 5,398 identical matches). For seed-only re-alignments, binned by two-sided unit-set Jaccard between a published group and its best-overlap match (`output/ws2/9b/matches_seedonly_jaccard.csv`):

   | unit Jaccard | n | ρ δ | ρ Λ | ρ p | same side of p=0.05 |
   |---|---|---|---|---|---|
   | 1 (identical) | 552 | 1.00 | 1.00 | 1.00 | 0.998 |
   | 0.9–1 | 196 | 0.96 | 0.78 | 0.77 | 0.81 |
   | 0.8–0.9 | 613 | 0.87 | 0.64 | 0.69 | 0.82 |
   | 0.6–0.8 | 1231 | 0.74 | 0.51 | 0.53 | 0.74 |
   | < 0.6 | 2500 | 0.55 | 0.14 | 0.19 | 0.62 |

   δ is robust to membership changes. Λ is not: with a median T of 8, adding or removing one or two centroids changes Σ̂ substantially. Carry-over is graded by evidence: across 10 seeds, published groups with p<0.001 have a p<0.05 match in 60% of runs, p 0.001–0.01 in 54%, 0.01–0.05 in 47%, 0.05–0.2 in 23%, >0.2 in 15% (base rate 26%). **WS1's 22 q<0.05 groups have a p<0.05 match in 62% of seed-only runs (64% across all UMAP/max_cluster_size variants).** By group: 856 and 563 10/10, 22 and 655 9/10, 4 and 646 8/10, **1002 7/10**, 821 5/10, **983 4/10**, 712 2/10.
6. **Qualitative check (matched by shared units; keywords read only as labels).** Matched groups share units, so their labels agree partly by construction. The evidence is in the units that *moved*. For a seed-only re-alignment, the units a matched group gains share top-10 keywords with the published group in **93%** of cases, and the other groups that take some of a published group's units do so in **92%**, against **5%** for a random group. It is the same at n_neighbors = 50 (91% / 93%) and max_cluster_size = 20 (91% / 93%), and lower but far above chance at min_cluster_size = 10 (79%) and with no UMAP (75% / 97%). Examples from seed 1: 1002 `clinton hillary trump campaign bernie` gains `trump democratic election clinton campaign` and loses units to `obama mccain campaign`; 221 `music listen track song` splits 4 ways into `youtube music song | music listen album | music remix youtube`; 1001 `president trump obama` splits 6 ways, including `impeachment …` and `republican president …`; 798 `scientific evolution science` gains `belief conclusion theory science`. One counter-example is worth showing: 290 "conspiracy" gains units labelled `rich headline bless taxes business`. **So the disagreement is about boundaries and granularity, not about which topics exist** — WS3's conclusion, reached here on fixed units with the encoder held constant.

*Caveats.* Agreement is unit-weighted, so it cannot be compared directly with WS3's comment-weighted arm A numbers; it can be compared with arm B2, and the seed-0 control is identical. The variants are one-at-a-time around the published setting, not a factorial. The keyword Jaccard uses the published per-cluster TF-IDF, which for 2018-03 may describe a stale partition (Cross-cutting issues).

Plan (original): Cheap: 10,200 × 384 centroids. Grid over UMAP n_neighbors {10,15,30,50}, min_dist {0,0.1,0.25}, align_dim {5,10,20}, HDBSCAN min_cluster_size {3,5,10}, × 10 seeds. Report ARI/AMI against the published partition, the number of groups, the noise fraction, and stability of per-group drift for the Table 1 topics. Add a no-UMAP baseline (e.g. agglomerative or HDBSCAN on cosine distance between centroids).

**Response.** We thank the reviewer for this suggestion. We have added a sensitivity analysis of the alignment step (new S4 Appendix and S2 Table). Because UMAP and HDBSCAN operate only on the 10,200 monthly topic centroids, every variant relabels exactly the same units, so alternative groupings can be compared with the published one directly, with no matching heuristic. We re-ran the alignment with 10 random seeds at the published settings, and with each parameter varied in turn around them: UMAP n_neighbors (10–50), min_dist (0–0.25) and output dimension (5–20), and HDBSCAN min_cluster_size (3–10), min_samples (1–5) and max_cluster_size (20, 50, 100, unbounded). For each grouping we re-ran the drift test of §2.6 on every group, and we added a baseline without UMAP (HDBSCAN applied directly to the centroids, on cosine distance and after PCA to 50 dimensions).

The findings are as follows.
1. **The drift result does not depend on the alignment settings.** Across all seeds and all UMAP and max_cluster_size settings, the share of tested groups with significant time-dependence (p < 0.05) lies between 25% and 27%, against 26.3% for the published alignment (9–11% at p < 0.01, against 9.8%).
2. **Changing the UMAP parameters perturbs the grouping no more than changing the random seed.** Agreement with the published grouping, measured on centroids that both groupings assign to a group, is ARI 0.61–0.69 across the UMAP settings, against 0.67–0.73 across seeds alone. HDBSCAN's minimum group size is the one consequential choice. It sets the granularity of the topic groups: larger minimums give fewer, longer-lived groups and, with longer trajectories, more power to detect time-dependence. We now describe this choice in §2.5.
3. **The dimensionality reduction is necessary.** Without UMAP, HDBSCAN leaves 65–69% of centroids unassigned, against 28% with it.
4. **Individual groups are less stable than the aggregate.** A different seed moves some centroids between neighbouring groups. Since the test statistic depends on which centroids make up a group, a group's p-value can change when its membership does. Of the groups significant after FDR correction in the revised Table 1, a best-matching counterpart remains significant (p < 0.05) in 62% of re-alignments on average. The chance rate is 26%, and the share increases with the strength of the original evidence. We therefore report this proportion for each group in Table 1 and interpret individual groups with corresponding caution (§3.2).
5. **The groups change at their boundaries, not in what they are about.** Matching groups by shared centroids, with keywords read only afterwards as labels, the centroids that move between groups under a different seed carry the topic of the group they left. Their keywords overlap that group's in 92–93% of cases, against 5% for a random group. For example, under one alternative seed the "clinton hillary trump" group gains centroids labelled "trump democratic election clinton campaign" and cedes others to an "obama mccain campaign" group, and the "music listen track" group divides into "youtube music song", "music listen album" and "music remix youtube". The alignment's uncertainty is about where one topic group ends and the next begins, not about which topics exist.

We also corrected the description of the alignment: it now states every parameter value and notes that the published alignment was not seeded (the revised analysis fixes all seeds).

**Manuscript changes.** *(Draft. Wording that depends on the revised Table 1 is marked [WS1/WS4]. The S-numbering assumes no other new appendix is inserted before it.)*
- **§2.5, l.152–154** — Old: "We specifically apply the Uniform Manifold Approximation and Projection (UMAP) algorithm to reduce the centroid embeddings to d = 10 dimensions, followed by Hierarchical DBSCAN (HDBSCAN) clustering to group temporally related topics." → New: "We specifically apply the Uniform Manifold Approximation and Projection (UMAP) algorithm (Euclidean metric, 15 neighbours, minimum distance 0.1) to reduce the centroid embeddings to d = 10 dimensions, followed by Hierarchical DBSCAN (HDBSCAN; minimum cluster size 3, maximum cluster size 100) to group temporally related topics. Centroids that HDBSCAN does not assign to any group (28%) are treated as transient topics and excluded from the drift analysis. The minimum cluster size sets the granularity of the topic groups. S4 Appendix shows that the aggregate drift results are insensitive to the UMAP parameters, the random seed and the maximum cluster size, and that group-level results should be read with the caveats given there."
- **§3.2** [WS1/WS4], after the revised Table 1 is introduced — New: "Because the alignment step can move individual centroids between neighbouring groups, we also report, for each group, the proportion of ten re-alignments with different random seeds in which its best-matching group remains significant (S4 Appendix). Aggregate rates of significant time-dependence are stable to within one percentage point across these re-alignments."
- **§4.1, l.263–268** [WS4, which also owns the "Appendix 4.2" → S2 fix] — Old: "Another limitation arises from the use of non-deterministic unsupervised clustering methods, … This is both computationally expensive as well as challenging to implement." → New: "Another limitation arises from the use of non-deterministic unsupervised methods in both the clustering and the alignment stages. We assess the first in S2 Appendix. For the second, S4 Appendix shows that the share of topic groups with significant time-dependence is insensitive to the UMAP parameters and random seed, but that the boundaries of individual groups are not. A given group is best read as one realisation of a topic whose core is stable and whose edges are uncertain."
- **New S4 Appendix, "Sensitivity of the topic alignment"** — a short text covering the design (10 seeds; one parameter at a time; no-UMAP baseline; agreement on units both groupings assign, with noise reported separately; drift re-tested on each grouping with B = 10⁵) and findings 1–5 above, with the per-unit-Jaccard table from the [internal] block showing that δ is robust to membership changes and Λ less so.
- **New S2 Table** — `tables/R2.4_grid_summary_draft.md`, trimmed to: variant, groups, noise, ARI vs published, drift rate p<0.05 and p<0.01, ρ δ, ρ Λ.

**Figures / tables.**
- [`tables/R2.4_grid_summary_draft.md`](tables/R2.4_grid_summary_draft.md) → S2 Table (needs trimming). Per-run detail: `tables/R2.4_grid_runs_draft.csv`.
- [`tables/R2.4_baseline_draft.csv`](tables/R2.4_baseline_draft.csv): summary of the published grouping and its seed-0 re-alignment.
- [`tables/R2.4_keyword_summary_draft.csv`](tables/R2.4_keyword_summary_draft.csv), [`tables/R2.4_keyword_examples_draft.csv`](tables/R2.4_keyword_examples_draft.csv): the qualitative check.
- *Not built yet, optional:* a one-panel S-figure plotting the drift rate (p<0.05) and ARI against the published grouping per variant, with seed ranges — the table carries the same information.
- *For WS1:* the per-group "significant in N/10 re-alignments" column for the revised Table 1 can be generated from `output/ws2/9b/matches.csv`.

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

**[internal]** Workstream: WS3 · Status: **main matrix complete (2026-10-27); n-ladder control running; write-up drafted below.** The 6-window × 3-null matrix ran in 1.7 h on prospero (`src/10k_nullrun.py --seed 0`); results below and in `refs/revision/tables/R2.8_nullmatrix.md`.  
Plan (agreed 2026-09-22, now largely executed):
- ~~**Now:** fix the resampling so it is a true bootstrap or random-subsample resampling.~~ **Done** — `src/10k_nullrun.py` implements all three modes (`init` / `data` / `both`); `5a_dist` untouched. It runs locally off the cached R2.1 subsample, so no cm job and no manifest was needed.
- ~~**After WS1 is complete:** apply the corrected WS1 test to each null run.~~ **Done** — six windows (2007-02, 2010-02, 2015-01, 2017-03, 2020-06, 2021-06) × three null modes, false-positive rate reported per window before and after BH.
- **Remaining:** the n-ladder control (running 2026-10-27), the R ladder (proposed below, not started), and applying the S2 rewrite in Overleaf.

**Audit of the published validation (2026-10-27).** Three findings, all from reading S2 against the code and the stored runs. The first two change what the revision has to say; the third constrains how any new run must be configured.

*1. S2 describes a resampling the code does not do.* S2: "resampled it with replacement R times within batches of size M". The code (`5a_dist_singlemonth.py:121`) calls `np.random.permutation` — **without** replacement — and the permutation has no effect in any case: a single `MiniBatchKMeans.partial_fit` updates centres as a weighted mean over assigned points, which is order-invariant. The only quantity that differs between resamples is the k-means++ initialization, because `kmeans_plusplus(batch[idx,:], random_state=KMEANS_SEED)` draws from permuted rows at a fixed seed. The code's own comment one line above the permutation also says "sampled with replacement", so S2 inherited the error from the comment.

**The S3 Fig caption already describes this correctly** — "using a single shared centroid initialization in each permutation **(a)** and different initialization **(b)**" — while the S2 body attributes the difference between (a) and (b) to batching (single batch vs M = 1M). The caption matches the code; the body does not. The revision should make the body agree with the caption and say plainly that the published experiment varies the initialization, not the data.

*2. None of the stored null runs used the published alignment.* Audited every `align_model_*.pkl` under `km/kmp`, `km/kmp50` and `mbkm_50/bootstrap`:

| run | `min_cluster_size` | `max_cluster_size` |
|---|---|---|
| **published `mbkm_50/align/max_100`** | **3** | **100** |
| every null run | 5 | 300 |
| `mbkm_50/bootstrap/2007-02` | 5 | 205 |
| `mbkm_50/bootstrap/2007-05` | **50** | 300 |

So **S2 does not validate the pipeline as published**, and G is not comparable across those runs (2007-05's `min_cluster_size=50` in particular makes its G = 33 / median T = 117 an artefact of the setting, not of the window). New runs use `wc.HDBSCAN_KW`, i.e. the published 3 / 100.

*3. G depends strongly on R, so R must match the analysis it validates.* S2 predicts G ≈ k and reports G = 50 (R = 20, deterministic) and G = 49 (R = 20, batched). The stored runs bear that out **at R = 20** — G = 48–73 over six windows with median T ≈ R — and break it badly at R = 204, where G = 339–408 with median T = 13. The cap is not the cause (median T is far below any of the caps); the likely mechanism is UMAP, where `n_neighbors=15` over 10,200 points resolves much finer structure than over 1,000. **A null at R = 20 therefore cannot be read as validating a 204-period analysis**, and `10k_nullrun.py` defaults to `--n_resamples 204`.

*What the existing runs already give us, with no new compute.* WS1's R1.1 step 6 applied the corrected statistic to twelve of these runs — 2007-02/04/05/06/07, 2010-02, 2015-01, 2017-02/03/04, 2020-06, 2021-06 — spanning early, middle and late. Pooled over **1,405 tested groups the test rejects 4.84% at α = 0.05 and 0.78% at α = 0.01**, with per-run rates from 1.6% to 8.9%. (1,405 is the sum over runs, not one window: each run contributes G groups, and G ranges from 29 at R = 10 to 408 at R = 204.) Three further runs were excluded for cause: `km/kmp/2017-01` and `km/kmp50/2007-03` are warm-started chains — each resample initialised from the previous, so they are genuinely drifting sequences and rejecting is correct — and `km/kmp/2007-04` reproduces an identical solution every resample, which is S2 Fig S3(a)'s deterministic case by construction.

**This already answers the reviewer's literal request** (several early, middle and late windows under the existing method), subject to finding 2: those runs do not use the published alignment. The new runs below address that, and separate initialization noise from data resampling, which S2's claim requires.

**New runs: `src/10k_nullrun.py`** (written 2026-10-27, not yet run at full scale). Reads R2.1's cached uniform 200k/month MiniLM sample, so no full-month read is needed; the permutation test is exact under exchangeability at any n, and smaller n means noisier centroids, i.e. a harder test. `--nulls` separates the two sources of randomness that S2 conflates:

| mode | data | initialization | what it is |
|---|---|---|---|
| `init` | fixed | re-seeded each resample | reproduces the published experiment |
| `data` | bootstrap **with replacement** | one shared fixed `C0` | what S2 claims to do |
| `both` | bootstrap | re-seeded | the realistic case |

Defaults: `--n_resamples 204`, published alignment (`wc.UMAP_KW` / `wc.HDBSCAN_KW`), `--n_perm 100000`. `--ladder` repeats a mode at several n to show the rejection rate is flat in n. `--tag` keeps a second run's tables separate. Cost measured at **7.4 min per (window × mode)**, so the 6 × 3 matrix is ~2.2 h and the ladder control ~1.5 h, both local on prospero.

`align_units` and `drift_table` moved from `10g_compare.py` into `ws3_common.py` so R2.1 and R2.8 share one implementation (lazy `umap`/`drift` imports, so `10c`–`10e` still import cleanly in cm pods). Verified no change: `R2.1_armB2_grouping_draft.csv` is byte-identical after the move.

**Full matrix (2026-10-27): six windows × three null modes, 11,471 tested groups.** `src/10k_nullrun.py --seed 0` — R = 204 surrogate periods, k = 50, n = 200k uniform rows per window (2007-02 is the whole month, 68,511), published alignment, B = 10⁵. 1.7 h on prospero. Per-run table in `refs/revision/tables/R2.8_nullmatrix.md`; raw rows in `R2.8_nullrun_summary_draft.csv` and `R2.8_nullrun_groups_draft.csv`; parameters in `R2.8_params.json`.

*Result 1 — the significance procedure is calibrated, and the claim is stronger than a rejection rate.*

| mode | G | noise frac | tested | rej p<.05 | rej p<.01 | BH q<.05 | median δ |
|---|---|---|---|---|---|---|---|
| `init` (data fixed, re-seeded) | 1013–1092 | 0.243 | 3,838 | 0.0433 | 0.0086 | 0 | 0.093 |
| `data` (bootstrap, shared `C0`) | 975–1046 | 0.259 | 3,817 | 0.0469 | 0.0094 | 1 | 0.039 |
| `both` | 1001–1062 | 0.249 | 3,816 | 0.0493 | 0.0063 | 0 | 0.098 |
| **pooled (18 runs)** | **975–1092** | **0.235–0.270** | **11,471** | **0.0465** (95% CI .0427–.0505) | **0.0081** | **1** | — |

The useful form of this is not "we found few false positives" but **the whole null p-distribution is uniform**: KS against U[0,1] gives D = 0.0113, p = 0.107 pooled, and p = 0.31 / 0.40 / 0.61 for `data` / `both` / `init` separately, with mean p = 0.4991. Benjamini–Hochberg yielded **one** survivor across 11,471 tests. Per-window rejection ranges 3.2–6.8%. Against that, the real published run rejects 155 of 589 (26.3%) with 19 surviving BH, at median λ 0.541 vs 0.18 and median δ 0.183 vs 0.093; a binomial test of 155/589 against the pooled null rate gives p = 3e-70. **This is the answer to R2.8, and it is a clean pass in every window and every mode.**

*Result 2 — G, the noise fraction and the lifespan distribution are properties of the alignment configuration, not findings about the corpus.*

| | null (18 runs) | real published run |
|---|---|---|
| groups G | 975–1092 (mean 1,031, sd 34, CV 3.2%) | **1,006** |
| noise fraction | 0.235–0.270 | 0.279 |
| tested (T ≥ 5) | 606–662 | 589 |
| median T | 7–8 (q25 6, q75 11, max 82–98) | **8** |
| median δ | 0.034–0.111 | 0.183 |

The six windows span monthly comment volumes from 68,511 (2007-02) to 185,204,658 (2021-06) — a 2,700× range in how much topical material is available — and G does not move. 2007-02 is additionally an unplanned n-control: its "uniform 200k draw" is the entire month at 68,511 rows, and it gives G = 975–1034 against 977–1092 for the five 200k windows, so G is already flat across a 3× change in n before the ladder arm reports. This supersedes an earlier reading of the *old* runs (G = 339–408, median T = 13) that suggested real data fragments ~2.7× more than the null; that gap was entirely an artefact of the mismatched HDBSCAN settings in finding 2 above.

*Result 3 — the test is invariant to displacement scale, which is why all three modes calibrate alike.* `data` mode gives median δ = 0.039 against `init`'s 0.093, a factor of 2.4 **lower** — bootstrapping 200k rows moves centroids *less* than re-seeding k-means++ does, because every bootstrap resample starts from the same shared `C0` and converges near the same local optimum while re-seeded runs land in different ones. All three modes nevertheless reject at the nominal rate, because Λ compares a trajectory against permutations of its own steps, so the absolute step scale cancels. Worth a sentence in S2: calibration does not depend on how large the null perturbation is. It also means S2's *claimed* design (bootstrap with replacement) is the **weaker** null on displacement magnitude, not the stronger one.

**What this does not establish.** Three limits, to keep the write-up inside the evidence:

1. *The mechanism behind G ≈ 1,000 is unknown.* Working hypothesis, untested: HDBSCAN returns clusters near the smallest size it is permitted — ~7,650 non-noise units over ~1,031 groups is a mean group size of 7.4 against `min_cluster_size=3` — at a resolution scale set by UMAP's `n_neighbors=15` over 10,200 points. Settling it needs a settings sweep on real and null input together, which R2.8 does not require and which we should not start for this revision.
2. *S2's reported G = 50 is still unexplained at the published settings.* Every run in this matrix is R = 204; the published S2 got G = 50 at R = 20, and the stored runs give G = 48–73 at R = 20 under the *mismatched* 5/300 settings of finding 2. So S2's apparent agreement with its own G ≈ k prediction is plausibly an R = 20 artefact rather than a confirmation — but that has not been shown at 3/100. **Proposed next run (flagged, not started): an R ladder** on one window, `init`, published settings, R ∈ {20, 50, 102, 204}, ~25 min. It reproduces or refutes G = 50 at R = 20 and gives the G-vs-R curve, which is both the cheapest evidence toward item 1 and the thing a reviewer will notice before we do.
3. *"The null reproduces G" is not "the groups are uninformative",* and the two must not be conflated. The null's groups cannot carry temporal information by construction — every unit comes from one month. What matches is the *count and size statistics* of the grouping, which are geometric properties of UMAP + HDBSCAN on 10,200 points. Group *content* is a separate question and arm B2 answers it the other way: real groups show 75× keyword coherence with 99% of groups above the shuffle null. The supportable statement is narrow — **G is not evidence about the number of topics in the corpus** — and nothing here says the grouping is noise.

G is not load-bearing in the paper (user, 2026-10-27), so finding 2 is a wording repair rather than a result that needs defending. The thing to avoid is a sentence that reads G or the median lifespan as an estimate about Reddit.

**Response.** (draft 2026-10-27)

> We have substantially expanded this analysis and we thank the reviewer for prompting it. We now report the resampling null at six time windows spanning the corpus — 2007-02, 2010-02, 2015-01, 2017-03, 2020-06 and 2021-06, covering early, middle and late periods — and at each window under three resampling schemes that separate the two stochastic elements of the clustering step: re-drawing the k-means initialisation on fixed data, bootstrap-resampling the month's comments with replacement under a fixed initialisation, and both together. Each run passes R = 204 surrogate periods, matching the 204 months of the main analysis, through the complete pipeline: clustering, UMAP and HDBSCAN alignment at the settings used for the published results, and the drift test with 10⁵ permutations. Because every surrogate period is drawn from the same month, any displacement the pipeline recovers must be an artefact of the method.
>
> Across the eighteen runs the drift test examined 11,471 groups built from data containing no temporal signal. It rejected 4.65% of them at α = 0.05 (95% CI 4.27–5.05) and 0.81% at α = 0.01, with per-window rates between 3.2% and 6.8%, and the full distribution of null p-values is indistinguishable from uniform (Kolmogorov–Smirnov D = 0.011, p = 0.11). After Benjamini–Hochberg correction, one group out of 11,471 survived at q < 0.05. The corresponding figures for the real corpus are 26.3% of 589 groups at p < 0.05 and 19 surviving correction, at twice the median displacement. The calibration holds equally under all three resampling schemes, which differ by a factor of 2.4 in the size of the perturbation they introduce; this is expected, since the test statistic compares a trajectory against permutations of its own increments and is therefore invariant to the scale of those increments. These results are reported in full in S2 Table [new].
>
> The re-examination also corrected two errors in the original S2. First, the text described the resampling as drawing "with replacement", whereas the code permutes without replacement — and because minibatch centroid updates are order-invariant within a batch, the permutation has no effect, so the published experiment in fact varied the k-means initialisation rather than the data. (The S3 Fig caption described this correctly; the body did not.) We have corrected the description and, rather than only restating it, added the two genuinely stronger nulls above. Second, S2 stated that aligned groups recovered from a single resampled window should form "an exact set of G = k groups". That expectation is incorrect at the number of periods and alignment settings used in the main analysis: a signal-free null yields 975 to 1,092 groups with a median lifespan of 7–8 periods, closely matching the 1,006 groups and median lifespan of 8 months obtained from the corpus itself, and G is flat across windows whose monthly comment volumes differ by more than three orders of magnitude. We now state this explicitly — the number of aligned groups and their lifespan distribution are properties of the alignment procedure at a given number of periods, not quantities estimated from the corpus, and we do not interpret them as such. We note that this does not bear on whether the groups are substantively coherent, which we assess separately [cross-ref R2.1], and it does not affect any result in the paper, none of which depends on the value of G. What the validation does establish, and what the analysis relies on, is that the drift test applied to these groups has its nominal false-positive rate at every window we examined.

**Manuscript changes.** All in S2 Appendix unless noted. The user applies these by hand; line references are to the verified fragments quoted above, not to Overleaf line numbers.

1. **Replace the resampling description.** Current: "we considered a single time window of data from a central portion of the entire 17-year corpus, resampled it with replacement R times within batches of size M". Proposed:

   > To assess whether the clustering and alignment steps can manufacture apparent drift, we took a single month of comments and generated R surrogate periods from it, so that any displacement the pipeline recovers must be an artefact of the method rather than a change in discourse. We did this under three schemes, which separate the two stochastic elements of the clustering step: (i) holding the data fixed and re-drawing the k-means++ initialisation for each surrogate period; (ii) bootstrap-resampling the month's comments with replacement for each surrogate period while holding the initialisation fixed; and (iii) both. Scheme (i) corresponds to the experiment reported in our original submission. Each resulting set of R × k surrogate centroids was then passed through the same UMAP and HDBSCAN alignment, and the same drift test, as the main analysis.

2. **Replace the G ≈ k expectation.** Current: "we would expect ... the resulting aligned groups to be an exact set of G = k groups". Proposed:

   > The number of aligned groups G recovered from such a surrogate set is not k. Across six windows and three schemes G ranged from 975 to 1,092 (mean 1,031), with a noise fraction of 0.24–0.27 and a median group lifespan of 7–8 periods — closely matching the 1,006 groups, noise fraction of 0.28 and median lifespan of 8 months obtained from the real corpus. The six windows span monthly comment volumes from 6.9 × 10⁴ to 1.9 × 10⁸, and G is flat across that range. G is therefore determined by the alignment procedure at a given number of periods R rather than estimated from the data, and we do not interpret it as a count of topics; no result in this paper depends on its value. Fixing R at 204, the number of periods in the main analysis, is necessary for the surrogate runs to be comparable to it.

3. **Make the S2 body agree with the S3 Fig caption.** The body attributes the difference between panels (a) and (b) to batching (single batch vs M = 1M); the caption correctly attributes it to shared vs different centroid initialisation. Change the body to match the caption.

4. **Add the new results**, as a paragraph reporting the pooled rejection rate, the KS uniformity test, the single BH survivor and the per-window range, plus one sentence on scale invariance (Result 3) and a new **S2 Table** carrying the 18-row matrix (`refs/revision/tables/R2.8_nullmatrix.md`).

5. **Main-text check — not yet audited, flagged for the user.** Grep the manuscript for any sentence that reports the group count or median group lifespan as a *finding* ("1,006", "1006", "topic groups", "lifespan", "median duration", "persisted for"). Given finding 2, such sentences should describe them procedurally. The user has confirmed G is not load-bearing, so this is expected to be a small number of wording changes, not a substantive revision. Nothing here requires softening claims about drift, which is the part the null clears.

**Figures / tables.**
- `refs/revision/tables/R2.8_nullmatrix.md` — the 18-row matrix, formatted for S2 (new **S2 Table**). **Done.**
- `refs/revision/tables/R2.8_nullrun_summary_draft.csv`, `R2.8_nullrun_groups_draft.csv`, `R2.8_params.json` — raw outputs of `src/10k_nullrun.py --seed 0`. **Done**, drop the `_draft` suffix once the ladder arm confirms.
- *Proposed, not built:* a QQ plot of the 11,471 pooled null p-values against uniform with the real run's 589 overlaid. One panel, and it carries the whole R2.8 argument visually — the null on the diagonal, the real run bowed hard toward zero. Cheap: the p-values are already in `R2.8_nullrun_groups_draft.csv`.

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

**2026-10-20 (WS3 → WS4, R1.2 / §2.1).** Two small corrections to how the corpus filter is described, both from WS3's retrieval work (details under R2.1).
1. **Empty comment bodies are kept.** The filter tests `'body' in entry`, which is true for `body == ''`. Such comments pass and all embed to the same point, like `[removed]`. Rate ~1e-4 (2 of 20,430 sampled rows in 2006-06). Same artifact family as the `[removed]`/`[deleted]` finding already recorded in Cross-cutting issues; one clause in the §2.1 filter description covers it.
2. **The filter is now verified, not assumed.** Re-streaming a month with the published filter reproduces the stored row count exactly (2006-06 and 2008-06, against both `totals.csv` and `len(labels)`), the streamed comment id matches the metadata id for every checked row (600/600), and re-embedding retrieved text reproduces the stored embedding to cosine 1.00000. If the Methods need a sentence on reproducibility of the preprocessing, these are the numbers to use.

**2026-10-20 (WS3 → WS1, WS2, WS4; important).** **Four months of the published run were clustered on a chronological prefix of the month.** Found when `10c_sample.py` hit a labels/`totals.csv` mismatch; an audit of all 204 months followed.

| month | comments | labels | kept |
|---|---|---|---|
| 2020-06 | 167,658,460 | 154,000,000 | 91.9% |
| 2020-08 | 173,973,262 | 153,000,000 | 87.9% |
| **2020-09** | 162,790,357 | **12,000,000** | **7.4%** |
| 2022-06 | 198,770,867 | 108,000,000 | 54.3% |

*What "prefix" means.* The first X comments in file order — not a random subset of size X. The dumps are **strictly ordered by creation time** (verified 2026-10-20: `created_utc` is monotonically non-decreasing across 27 probes spanning 2020-09, from row 0 to row 12,000,001). So the 2020-09 prefix runs from **2020-09-01 00:00 to 2020-09-03 02:10 — the first 2.1 days of the month.** Estimated boundaries for the others, assuming a roughly even daily rate: 2020-06 through ~Jun 27, 2020-08 through ~Aug 27, 2022-06 through ~Jun 16. Exact dates come free from the option-2 pass below, which streams those months anyway.

*This is a bias problem, not a precision problem.* Even 2020-09's 12M rows give ~240k comments per cluster, i.e. a centroid standard error of √(0.84/240,000) ≈ 0.002 — 1.4% of WS1's median monthly step (0.139), so negligible. The entire concern is representativeness: that month's centroids describe the discourse of 1–3 September 2020 (one news cycle) rather than the month, and a systematic shift of that kind is exactly what a drift analysis is sensitive to. Reporting it as "fewer comments" would understate it.

*What happened.* Each truncation is at an exact 1,000,000-row chunk boundary, and the `tfidf` files' `sample_indices` (global row ids of the 100 comments nearest each centroid) stop at the same boundary — max index / labels length = 0.9994–1.0000 for these four, versus 1.0000 of the *full month* for controls. So the fit, the labels and the keywords all saw only the prefix: the clustering ran before `1_embed.py` had finished writing those zarrs. The embeddings are now complete, so the raw data is fine and only the derived cluster products are affected. This is **not** the known `2c_cluster.py` dropped-batch bug (that loses one partial batch, not 150M rows), and the 2020-09 prefix is roughly the first 2–3 days of the month.

*Impact (computed, not estimated).* 200 of 10,200 centroids are prefix-based, touching **97 of the 589 tested groups**; 30 of those have a prefix month at a trajectory endpoint. The telescoping identity limits the damage: δ_k = ‖c_last − c_first‖ depends *only* on the endpoints, so an interior prefix month leaves δ untouched and enters Λ only through Σ̂. Of WS1's **22 groups at q < 0.05, 7 are touched** (53 q=0.0029, 573 q=0.0069, 821 q=0.0212, 856 q=0.0212, 983 q=0.0214, 78 q=0.0240, 478 q=0.0477) but **only 983 has a prefix month as an endpoint** (last month 2020-08, 88% coverage — mild). Groups 53, 573 and 78 have an interior centroid from the 7.4% month, which is the worst case.

*Decision (user, 2026-10-20): option 2 first, then possibly option 3. Deferred — not blocking the R2.1 run.*

*Repair options, cheapest first.*
1. **Disclose only.** A sentence in the data description plus a line in the limitations. Defensible given the impact above, and it costs nothing.
2. **Re-estimate the four centroid sets without re-clustering** (WS3 can deliver this for free). Keep the published partition definition — nearest published centroid — but recompute each cluster's mean from a time-spread sample of the *complete* month. `10c_sample.py --sample_full_month` already does the sampling, and nearest-centroid assignment was verified to reproduce the published labels exactly (agreement 1.0000, 2026-10-19). Output: how far the prefix centroid sits from the full-month centroid, in units of WS1's median monthly step (0.139). If that distance is small, option 1 is clearly safe; if it is large, 983 and the three 2020-09 groups need a caveat or option 3.
3. **Recluster the four months** from the complete embeddings and re-align. Changes the published partition, so it would want re-running Table 1 and Fig 4, and it is really a WS1/WS2 decision rather than a WS3 one.

**WS2, two things for R2.2.** (a) The `km/km{30,50,70,90}` labels are complete for all four months, so the k-grid is *not* affected and the k=50 replicate is a genuinely independent full-month run — which also makes it a natural reference for how much the prefix matters. (b) That means a k-grid comparison on those four months partly confounds "different k" with "prefix vs full month"; worth excluding them or reporting both ways.

**WS3's own handling.** `10c_sample.py` no longer errors on the mismatch. By default it samples the labelled prefix, which keeps the R2.1 arms consistent with the published partition, and records `labels_truncated`, `n_rows_labels` and `n_beyond_labels` in the per-month params and in `summary.csv`. With `--sample_full_month` the uniform draw spans the whole month and rows past the prefix carry cluster −1 plus a `BEYOND` flag, to be labelled later by nearest centroid — that is the mode option 2 above uses.

**2026-10-20 (WS3 → WS1, WS2, WS4; second data defect).** **For 2018-03, the published `labels` and `model_cc` are inconsistent: they come from two different fits.** Found by the per-month certificate during the R2.1 retrieval run (73 of 204 months done at the time; 2018-03 is the only month so far below 0.995 agreement).

*Evidence.* Using the **stored** embeddings (so this is independent of WS3's retrieval): each row's nearest centroid in `model_cc_2018-03.npz` equals `labels_2018-03.npz` for only **4.0%** of rows (chance 2%). Not a cluster renumbering — optimal-matching purity is 0.49 with 43/50 distinct targets and only 2/50 contingency rows dominant. Not a file mix-up with a neighbouring month either (best-matching purity for cc from 2018-01…05 is 0.40–0.47, with 2018-03's own cc no better than its neighbours'). The label-group means sit 0.49 from the corresponding centroid, larger than the typical centroid norm of 0.42, and only 3/50 label-group means have their own centroid as nearest. Purity 0.49 is far above chance, so `labels` is a *valid but different* k=50 clustering of the same month, not corruption.

*WS3's retrieval is not implicated:* for the same rows, re-embedded retrieved text matches the stored embeddings at cosine 1.00000 (min and median), and stored-row and our-text nearest-centroid assignments agree with each other 1.0000. Both disagree with `labels` identically.

*Likely cause, and why it matters which file is stale.* `2a_cluster_s.py` writes in the order fit → `model_cc` → labels → `labels` → tfidf. A re-run that wrote the new `model_cc` and then died before rewriting labels would leave **new centroids with stale labels and stale tfidf** — exactly this signature. If that is what happened, the published centroids are a valid full-month fit and **the paper's results, which flow through `model_cc` → `3_align.py` → groups → drift, are unaffected**; what is stale is the labels file and, more importantly, **the TF-IDF keywords for 2018-03's 50 clusters, which would then describe the old partition**. Since group names come from member months' keywords, any group with a 2018-03 member could carry a mis-sourced keyword contribution. The opposite case (stale `model_cc`) would be more serious. **Open: one test settles it** — take the `tfidf` top-100 `sample_indices` for a few clusters and check whether those rows' stored embeddings are nearest to their own centroid. Consistent ⇒ tfidf matches cc and only `labels` is stale; inconsistent ⇒ labels and tfidf are the matched pair and `model_cc` is the odd one out. Deferred until the retrieval run finishes, since it needs scattered reads across a full month and would compete for sshfs bandwidth.

*Design consequence for WS3, applies to every month.* **Cluster membership will be defined by nearest published centroid, not by the `labels` file.** That is the assignment rule the published centroids imply, it reproduces `labels` at ≥0.995 for every other month checked, and it makes arm B correct for 2018-03 without special-casing. Arm A (which re-clusters the uniform sample) is unaffected either way. One residual: 2018-03's *stratified* draw was made from the stale labels, so its per-cluster counts under the centroid partition will be uneven — cheap to fix by resampling and re-retrieving that single month (~25 min) if arm B needs balanced n there.

*Incidental benefit for R2.1.* Purity 0.49 between two independent k=50 MBKM fits of the same month is a direct, real-data measurement of the **within-model seed-instability floor** that arm A needs as its reference scale. Cross-model ARI should be read against that, not against 1.0.

**2026-10-20 (WS3, operational — OOM during the R2.1 retrieval).** The retrieval run killed the box at 15:21 after 90 of 204 months. Cause, from `journalctl -b 0 -k`: `Out of memory: Killed process 19893 (python) total-vm:24292528kB, anon-rss:15377308kB` — one worker at **15.4 GB resident** triggered a global OOM. No kernel panic and no reboot; the OOM kill took down the GNOME session, which is what made it look like a restart.

*Cause.* `ws3_common.load_metadata_ids` used `pickle.load` on the whole `metadata2` file to look up ~400 comment ids for the certificate's id check. Those pickles are a list of ~n four-element Python lists: 1.2–2.4 GB on disk for 2013–2015, expanding to roughly 6× that in objects. The run survived the 86 no-metadata months (2015-11+), completed four metadata2 months, then died with six workers simultaneously in the 2014 pickles. The `metadata/` gzip CSVs (2006-01 … 2012-12) were never a risk — those are streamed, O(len(rows)).

*Fixes.* `load_metadata_ids` takes `max_bytes` and raises `MetadataTooLarge` without loading; `10d_retrieve.py --metadata_max_gb` (default 0.5) catches it and records `id_check: {skipped: metadata2 too large}`. The nearest-centroid check still covers those months and is the stronger test anyway. New `--verify_only` re-runs the checks from text already on disk with no dump pass, so the id check can be added later for the big months one at a time (`--verify_only --workers 1 --metadata_max_gb 4`, ~45 min for the 30 affected months; optional).

*Nothing was corrupted.* All 90 completed months passed a cert/text integrity check (certificate present, text size matching, `complete: true`) — the certificate is written last precisely so a killed month is redone rather than trusted. Six orphaned `.part` files were removed.

*Measured footprint after the fix:* ~1.0 GB peak RSS per worker, dominated by torch + MiniLM, so 12 workers ≈ 13 GB on a 62 GB machine. The 114 remaining months are all bz2 (no zstd window to allocate; the Reddit zst files need `max_window_size=2**31`, up to 2 GB per stream, which is worth remembering if a future job runs many zst months at once).

**2026-10-21 (WS3 → all, R2.1 retrieval complete + a `metadata2` correction).** All 204 months retrieved: **54,880,577 rows, 5.25 GB of gzipped text**, every certificate present and `complete`, and every month's retrieved count equal to its sampled count.

*The certificates double as an audit of the published run.* Nearest-centroid agreement (re-embedded retrieved text → nearest published centroid, vs the published label) over all 204 months: **median 1.0000, mean 0.9947, 203 months ≥ 0.99**. The single exception is **2018-03 at 0.0350**, already documented — so that labels/centroids defect is confirmed isolated to one month out of 204.

*A correction to the `metadata2` fact (details in Verified shared facts).* The id check initially failed for every sampled row of 2015-07, 2015-08 and 2015-10 while those months' nearest-centroid agreement was ≥ 0.99 — i.e. the check was wrong, not the retrieval. Cause: `load_metadata_ids` keyed on the pickle record's first field as the row index, which holds in 2007 but is the 1M-row batch counter from 2015-07 on. Keying on list position gives 300/300 for both eras. Fixed in `ws3_common.py`. **Anyone using `metadata2` should index by position, not by the first field.**

*Final id-check tally:* 87 months run, 33,600/34,800 before the fix; the 1,200 "failures" were all from the three mis-keyed months and are now correct. 31 months were skipped by the memory guard and 86 have no metadata; all are covered by the nearest-centroid check, and the id check can be added for the skipped months with `10d_retrieve.py --verify_only --workers 1 --metadata_max_gb 4` if belt-and-braces is wanted.

**2026-10-26 (WS3 → WS2; `src/matching.py` is live, per your request).** The matching helpers are now a shared module. WS3 owns it, WS2 imports only; `10h_arma_inspect.py` and `10i_armA_summary.py` import from it and their local copies are deleted. No I/O, no `config.ini`, no paths — numpy/pandas/scipy/sklearn only. Every docstring carries the measure's failure mode, copied from R2.1's "Measures used, with failure modes" so the module and this document can't drift apart.

| function | signature | returns |
|---|---|---|
| `matched_purity` | `(a, b, k)` | `(purity, k×k contingency)` |
| `pair_scores` | `(a, b, rng=None, n_null=3)` | dict: `n, n_a, n_b, purity_1to1, overlap_many_to_one_median, ari, ami, homogeneity, completeness` (+ `*_null` keys iff `rng`) |
| `match_many_to_one` | `(a, b, weights=None, min_overlap=None)` | DataFrame: `g_a, g_b, overlap, n_a, n_a_core, frac_to_noise` |
| `noise_summary` | `(a, b)` | dict: `frac_noise_{a,b,either,both}, jaccard_noise, agreement_noise_or_not` |
| `containment` | `(sa, sb, mode='min')` | float |
| `enrichment_test` | `(hit, tot, p0)` | dict: `hit, tot, rate, chance_rate, enrichment, p` |

Self-test: `python 10z_matching_check.py` (24 assertions, <1 s, no data). It covers the identities, chance behaviour, the merge/split/all-to-noise example, weighting, and the two traps below.

**Four deviations from the API you specified. Flagged here rather than changed silently, as you asked — say so if any of them is wrong for R2.4/R2.2 and WS3 will adjust.**

1. **`matched_purity` now raises on labels outside `[0, k)` instead of accepting them.** As written it does `M[x, y] += 1` on a fixed k×k array, so a `-1` wraps into row/column k−1 and is silently counted as a real cluster. That never fired in R2.1 because it is only applied to k-means labels (0–49), but R2.4 compares HDBSCAN groupings where ~25% of units are noise, so it would have bitten you. **Use `pair_scores` for anything with noise or with differing group counts** — `matched_purity` assumes a square 1-1 matching over one shared k.
2. **`containment` takes `mode`.** Your spec is `|A∩B| / min(|A|,|B|)`, which is what R2.1 arm A reports and stays the default. But `mode='a'` (directional, `|A∩B| / |A|`) is needed to say *which way* a chain nests, and min() hides exactly that: arm B1's published set is 0.556 contained in minilm directionally while minilm is 0.939 contained in mpnet, and the symmetric form would have read both as "nested".
3. **`match_many_to_one` expects UNFILTERED labels, and returns `n_a_core` as well as `n_a`.** `frac_to_noise` and "overlap over non-noise members" cannot both hold on pre-filtered input, so the function filters internally: `overlap` is over members non-noise in both (R2.1's definition, unchanged), `n_a` counts all members, `n_a_core` is the `overlap` denominator, and `frac_to_noise = (n_a − n_a_core)/n_a`. **Note for your use:** 10h pre-filters to non-noise in *all three* spaces before matching, so its `frac_to_noise` is 0 by construction. Pass raw labels to get a meaningful value.
4. **`pair_scores` takes an optional `rng`.** Return keys are unchanged when it is omitted, as you required. Pass one and it appends `homogeneity_null`, `completeness_null`, `ari_null`, `ami_null` from a size-preserving shuffle. **Please use it.** Homogeneity and completeness are not chance-adjusted, and at ~1,000 groups over ~6,000 units the shuffle null is already ≈0.69 — reporting 0.90 as strong agreement without that baseline is wrong, and WS3 did exactly that earlier today before catching it.

**Two things WS3 got wrong on R2.1 that R2.4 will hit in the same form.** Both are now guarded in code, but they are judgement calls, not just API:

- **Never report an ARI over HDBSCAN output with noise included.** sklearn treats `-1` as a single cluster, which for us is ~2,800 of 10,200 units, so the score measures the noise boundary rather than the grouping. On arm B2 the all-units ARI read 0.11 where agreement on units both groupings place was 0.52 — we drew the wrong conclusion from it and had to revise a cross-workstream note. `pair_scores` drops noise pairwise; **always print `noise_summary` next to it**, because dropping noise flatters a variant that pushes hard-to-place units into noise instead of mis-grouping them. (`grouping_scores()` in `10g_compare.py` returns both views for the unit-level case; it stays in WS3 because it is tied to our arm B2 rungs, but copy the pattern.)
- **Don't compare BH-significant sets as sets.** At 589 tests q<0.05 membership is a knife-edge function of the whole p-distribution: our arm B2 has mpnet at 24 significant against minilm's 53 while holding *more* groups at both p<0.05 and p<0.01, on one BH rank. Compare **rate at a fixed p** and rank-correlate the statistic over all tested groups. Related, for any permutation testing: **use B = 10⁵**, per WS1's R1.1 note.

**No bugs were found in the extracted logic, and no R2.1 number changes.** Verified by re-running the two stages that use these functions and diffing against the pre-refactor outputs: `10i_armA_summary.py` → `R2.1_armA_{cluster_level,group_level,group_structure}.csv` all **byte-identical**; `10h_arma_inspect.py --stages grouplevel` → `R2.1_armA_group_agreement_draft.csv` **byte-identical**, and both `R2.1_armA_group_match_*_draft.csv` reproduce the same row count (573, 550), the same `g_b` for every group and `overlap` to machine precision (max |diff| 0.0e+00). `frac_to_noise` is 0 throughout, as expected from the three-way prefilter. One cosmetic change: `R2.1_armA_group_match_<base>_<t>_draft.csv` now has columns `g_a`/`g_b` where it had `g_base`/`g_other`, plus the two new columns. Final tables are untouched.

**2026-10-26 (WS3 → WS1, WS2, WS4; arm A results, two items that affect others).**

1. **Individual topic groups are only partly reproducible, and the alignment step is the weak link — but the earlier version of this note overstated it. Revised 2026-10-26, please re-read if you acted on it.** Two measurements, which answer different questions:
   - **Full pipeline, comment level (arm A, unchanged).** Re-running on identical data with only a different k-means seed reproduces individual groups at **median 22% overlap, 9.4% surviving at ≥50%**, 1-1 matched purity 0.181. Changing the encoder costs a further ~18–31%. This compounds cluster instability with grouping instability and stands as measured.
   - **Grouping step alone, over fixed units (arm B2, corrected).** The earlier note cited "grouping ARI 0.29 under the alignment seed alone". That figure included HDBSCAN noise, and sklearn's ARI treats `-1` as one cluster of ~2,800 units, so it was measuring the noise boundary more than the grouping. On units non-noise in both groupings it is **ARI 0.710 for the alignment seed alone and 0.477–0.521 for a different encoder** (AMI 0.68–0.85; chance ≈ 0). Substantial agreement, not near-zero.
   - **And the groupings agree on content more than on membership:** every re-derived grouping is *equally* keyword-coherent to the published one (median within-group term Jaccard 0.119–0.125 against a 0.0017 shuffle null, 99% of groups above), and tracing named published groups shows their members landing in destination groups with the same keywords — 221 "music" fragments into 6–8 groups in every space, all of them music groups.

   **So the honest claim is narrower than before:** group *boundaries and granularity* are unstable (each published group spreads over ~1.9–2.0 re-derived groups, symmetrically), and *which units are groupable at all* is the least stable thing (noise-set Jaccard 0.26–0.42). The groups are not arbitrary and they are not noise. **For WS1/WS2:** a sentence is still warranted wherever a specific group id carries weight, but it should say that group boundaries are sensitive to the alignment seed, **not** that the groups fail to reproduce. It does *not* undermine aggregate statements — group counts, noise fractions, span and T distributions are stable to a few percent across seeds and encoders — nor WS1's δ/Λ results, which arm B1 shows are highly rank-stable (δ Spearman 0.94–0.99). Relevant to R2.4 as much as R2.1.
   - **Directly for R2.4:** any ARI you compute over HDBSCAN output **must exclude noise on both sides** or it will report ~0.1 where the real agreement is ~0.5. `grouping_scores()` in `10g_compare.py` returns both figures plus the homogeneity null.
   - **New, and the best robustness result we have (arm B2 + drift):** re-grouping the same 10,200 clusters — by alignment seed, by centroid re-sampling, or by encoder — preserves the drift rate at **26–34% of tested groups at p<0.05** with matched median δ and T, even though the groupings share only about half their pairwise structure. The drift finding is a property of the corpus, not of MiniLM and not of our alignment choices. That is a stronger statement than B1 can make, because it conditions on nothing MiniLM selected.

2. **What drifts is mostly platform and media format, not topical discourse — now corroborated independently.** Arm A re-derives groups from scratch in three embedding spaces, and the strongest time-dependent groups are Digg→Reddit migration, browser technology, Reddit site features, moderation templates, YouTube/gif sharing, karma/repost meta and a datable meme era. That matches WS1's observation that most published groups surviving BH were platform/format/language groups. **For WS4 (R1.4, R2.9):** two independent analyses now support the cautious framing. The honest reading is that the measured drift largely tracks changes in the platform and in how content is shared, with covid the clearest substantive topical case. Caveat: this is a reading of keyword examples, not a quantified classification — see the caveats under R2.1.

Also for WS1, two things about the significance machinery, both found by repeating your mistakes:
- **Use B = 10⁵.** We ran arm B at 10⁴ first and reported q-counts from it, which is exactly the floor problem you documented under R1.1 (smallest attainable p 1.0×10⁻⁴ above BH's tightest threshold 8.5×10⁻⁵). Worth a line in the methods so no one repeats it a third time.
- **Don't compare BH-significant sets as sets.** Jaccard is confounded with set size, and containment is better, but the underlying problem is that q<0.05 membership at 589 tests is a knife-edge function of the whole p-distribution. Our arm B2 run has mpnet at 24 q<0.05 against minilm's 53 while holding *more* groups at p<0.05 and p<0.01 — one BH rank, a 29-group swing. **Calibration you may want to cite: even at B = 10⁵, our implementation and yours agree exactly on p<0.01 (58) and q<0.01 (6) but differ on q<0.05 (19 vs 22), ~15%, on RNG streams alone.** If the 22-group set is presented as a list, it is worth one sentence that membership near the boundary is not stable.

**2026-10-26 (coordination → WS2, WS3; the matching question, correcting the 2026-10-19 note).** The 2026-10-19 note said WS2 owns the cross-partition matcher and WS3 would consume it. **That is now stale: WS3 built matching at both levels while building arm A, and nothing is blocked on WS2.** WS2 should adopt WS3's procedure rather than write a second one.

*Two distinct levels, and they are not the same problem.*
- **Cluster level (within a month, k-means output).** `matched_purity(a, b, k)` in `10h_arma_inspect.py:58` — Hungarian 1-1 on the contingency table, plus ARI/AMI. Needs both labelings over the same point set.
- **Group level (whole dataset, HDBSCAN output).** `stage_grouplevel` in `10h_arma_inspect.py:324` maps each sampled comment → its (month, cluster) unit in each space → its group, then scores at comment level (ARI/AMI/homogeneity/completeness) and matches groups many-to-one **by shared comments**, with an `overlap` score, a `--min_overlap` threshold, and Spearman on δ/T plus containment on the significant sets. `pair_scores(a, b)` in `10i_armA_summary.py:54` gives the same battery for two labelings over identical units.

*The rules WS3 established, which WS2 should inherit:*
- **All matching is by shared members; TF-IDF keywords are labels only, never the matching key.**
- **Never match by centroid distance across embedding spaces** (384 vs 768 dims). Note this constraint does *not* bind WS2's within-space comparisons (k grid, alignment variants are all MiniLM 384-d), where centroid proximity is available and much cheaper than reading the large `labels` files. Membership matching is the more principled choice since clustering is about which comments group together; centroid matching is the cheap approximation. WS2 should pick deliberately and say which it used.
- **Measure selection and failure modes are already written up** under R2.1 ("Measures used, with failure modes"). Use containment, not Jaccard, when set sizes differ. Don't re-derive this.

*What WS2's three cases actually need — mostly less machinery than R2.1.*
- **R2.4 (alignment sensitivity): no matcher at all.** Every variant relabels the *same* 10,200 published centroids, so two runs are two labelings of identical units — `pair_scores` applies directly. This is the arm B2 trick.
- **R2.2 across k:** units genuinely differ (30 vs 50 clusters per month), so this needs either comment-level mapping (WS3's pattern; `labels` files are up to 220 MB/month, so subsample) or within-space centroid matching.
- **R2.2's k=50 replicate (`km/km50` vs `mbkm_50`):** same k and same months, so the unit *indices* coincide but the fits differ — cluster 7 is not the same cluster in both. Still needs matching, but it's within-space.

*Framing, and a correction to my own earlier wording.* I described the seed-sensitivity result to WS2 as "uncomfortable". That overstates it, and WS3's own write-up is more careful: ARI/matched purity **understate topical correspondence substantially** here, because with ~1000 groups almost every pair counts as "apart". The qualitative check is the persuasive part — matched one-to-one by member overlap, with keywords read only as labels, the matched clusters are recognisably the same topics even at ARI 0.18–0.26 (`deck dps ult decks buff` ↔ `deck dps ult mana crit`, 71%; `trump biden bernie president election` identical top-5 in both spaces). **The honest summary is that the partition is not a reproducible object while the topical structure it describes largely is: the disagreement is about which comments land where, not about which topics exist.** Matching clusterings is a genuinely hard measurement problem with no single right statistic, which is why WS3 reports a battery with stated failure modes. WS2 should carry that same framing into R2.4 rather than reporting a bare ARI.

**2026-10-26 (WS2 → WS1, WS4; R2.4 alignment sensitivity — what it means for named groups).** Full results under R2.4. Two findings for the text.

1. **The aggregate drift result does not depend on the alignment.** Over 10 UMAP seeds and every one-at-a-time variant of n_neighbors (10–50), min_dist (0–0.25), align_dim (5–20) and max_cluster_size (20, 50, none), the share of tested groups at p<0.05 stays at **0.25–0.27** (published 0.263) and at p<0.01 at 0.09–0.11 (published 0.098). Only HDBSCAN's granularity (min_cluster_size, min_samples) moves it: coarser groups are longer trajectories, so the test has more power. That is a resolution choice and should be described as one. This agrees with WS3's arm B2.
2. **Per-group significance is sensitive to group membership, and that is the alignment's doing, not the statistic's.** Λ is exactly reproducible for identical groups. But a seed-only re-alignment typically changes a group's membership (median two-sided unit Jaccard to its best match 0.60), and Λ is fragile to this at T ≈ 8: at Jaccard 0.8–0.9, ρ_Λ = 0.64 while ρ_δ = 0.87. **WS1's 22 q<0.05 groups keep a p<0.05 match in 62% of seed-only re-alignments**, graded by strength of evidence (60% for p<0.001 down to 15% for p>0.2; base rate 26%). 1002 holds in 7/10 seeds, 983 in 4/10, 821 in 5/10, and 856 and 563 in 10/10. **For WS4:** a sentence is needed wherever a named group's significance carries weight. Suggested: "group-level significance reflects one realisation of the alignment; under re-alignment with different seeds a group's best-matching counterpart remains significant in N of 10 runs". The ratio can go in the revised Table 1 as a column if WS1 wants it (WS2 can supply it from `output/ws2/9b/matches.csv`). **For WS1:** δ is much more robust to membership changes than Λ, which may bear on how the two are presented. The keyword evidence that the groups are nonetheless topically stable (units that change group stay on topic 92–93% of the time vs 5% by chance) is the qualitative counterweight and should go next to that sentence.

**2026-10-27 (WS3 → WS1, WS2, WS4; R2.8 — the number of aligned groups is a property of the alignment, not a count of topics).** Full results and table under R2.8. One narrow finding, plus an explicit statement of what it does *not* mean, because the available overstatement here is tempting and wrong.

**The finding.** Running one month's comments as R = 204 surrogate periods through the complete published pipeline — so the input contains no temporal signal whatsoever — reproduces the published grouping's count and size statistics almost exactly. Over six windows × three resampling schemes (18 runs): **G = 975–1,092** (mean 1,031, CV 3.2%) against the published **1,006**; noise fraction 0.235–0.270 against 0.279; tested groups 606–662 against 589; median lifespan 7–8 periods against 8. The six windows span monthly comment volumes from 68,511 to 185,204,658 — a 2,700× range in available topical material — and G does not move. 2007-02 is also a free n-control (the whole month is 68,511 rows, below the 200k target) and gives the same G.

**What to change:** any sentence that reports G, the group count, or the median group lifespan as a *finding about Reddit* should describe it procedurally instead. Candidates to check in your own text: "1,006 topic groups", "median lifespan of 8 months", anything of the form "we identify N topics". The value of G is set by UMAP + HDBSCAN at a given R, not estimated from the corpus.

**What this does NOT mean, and please don't propagate it as if it did.**
- **It does not say the groups are meaningless.** The null's groups *cannot* carry temporal information by construction — every unit comes from a single month. What matches is the count and size *statistics*, which are geometric properties of UMAP + HDBSCAN on 10,200 points. Group content is a separate question, and WS3's arm B2 answers it the other way: real groups show 75× keyword coherence with 99% of groups above a shuffle null. WS2's R2.4 agrees (units that change group stay on topic 92–93% of the time vs 5% by chance).
- **It does not weaken the drift result — it strengthens it.** The same 18 runs reject at **4.65%** pooled (95% CI 4.27–5.05, 11,471 tested groups) with **one** group surviving BH, and the null p-distribution is uniform (KS p = 0.11). The real run rejects 26.3% of 589 with 19 surviving. The pipeline manufactures the group *structure* but not the *drift*. Anyone writing about the test's validity now has a much stronger citation than the single-window check in the published S2.
- **It is not a reason to revisit a conclusion.** The user has confirmed G is not load-bearing in the paper (2026-10-27). This is a wording repair, scoped to the sentences listed above.

**Mechanism unknown.** We have a hypothesis (HDBSCAN returning clusters near its `min_cluster_size=3` floor at a resolution set by `n_neighbors=15` over 10,200 points) and no test of it. Don't assert a cause in your text. Note also that the published S2's reported G = 50 came from R = 20, and the G-vs-R dependence is not yet characterised at the published HDBSCAN settings — WS3 has an R ladder flagged for that.

**2026-10-27 (WS2 → all; k-grid provenance, corrects `km/readme.txt` and the "mixed 2014–15" wording in Verified shared facts).** From `avg_inertia` in each run's `stats_*.pkl` (valid ≈0.8 = full KM; ≈0.01 = MBKM last-batch artifact): **2014 is MBKM for every k** and **2015 is KM for km50 only** (MBKM for km30/70/90). The readme's "2014–2015 a mix" is true only across k for 2015. km30/50/70 are complete for all 204 months with labels matching `totals.csv` exactly; km90 lacks only 2014, which was never run, and WS2 is filling it on cm (`manifests/rev-ws2-km90-2014.yml` → `rev/ws2/km90/`). Relevant to R1.3's Fig S1 caption (2014–15 exclusion) and to anyone reading the k grid by era. Detail under R2.2.
