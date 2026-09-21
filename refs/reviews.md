Manuscript was submitted in Aug 2026 to PLoS One, returned 9/14/2026 requesting major revisions.

# From Academic Editor

Dear Dr. Morse,

Thank you for submitting your manuscript to PLOS One. After careful consideration, we feel that it has merit but does not fully meet PLOS One’s publication criteria as it currently stands. Therefore, we invite you to submit a revised version of the manuscript that addresses the points raised during the review process.

Please submit your revised manuscript by Nov 13 2026 11:59PM. If you will need more time than this to complete your revisions, please reply to this message or contact the journal office at plosone@plos.org. When you're ready to submit your revision, log on to https://www.editorialmanager.com/pone/ and select the 'Submissions Needing Revision' folder to locate your manuscript file.

Please include the following items when submitting your revised manuscript:
A letter that responds to each point raised by the academic editor and reviewer(s). You should upload this letter as a separate file labeled 'Response to Reviewers'.
A marked-up copy of your manuscript that highlights changes made to the original version. You should upload this as a separate file labeled 'Revised Manuscript with Track Changes'.
An unmarked version of your revised paper without tracked changes. You should upload this as a separate file labeled 'Manuscript'.
If you would like to make changes to your financial disclosure, please include your updated statement in your cover letter. Guidelines for resubmitting your figure files are available below the reviewer comments at the end of this letter.

If applicable, we recommend that you deposit your laboratory protocols in protocols.io to enhance the reproducibility of your results. Protocols.io assigns your protocol its own identifier (DOI) so that it can be cited independently in the future. For instructions see: https://journals.plos.org/plosone/s/submission-guidelines#loc-laboratory-protocols. Additionally, PLOS One offers an option for publishing peer-reviewed Lab Protocol articles, which describe protocols hosted on protocols.io. Read more information on sharing protocols at https://plos.org/protocols?utm_medium=editorial-email&utm_source=authorletters&utm_campaign=protocols.

As the corresponding author, your ORCID iD is verified in the submission system and will appear in the published article. PLOS supports the use of ORCID, and we encourage all coauthors to register for an ORCID iD and use it as well. Please encourage your coauthors to verify their ORCID iD within the submission system before final acceptance, as unverified ORCID iDs will not appear in the published article. Only the individual author can complete the verification step; PLOS staff cannot verify ORCID iDs on behalf of authors.

We look forward to receiving your revised manuscript.

Kind regards,

Ming Zhang
Academic Editor
PLOS One


# From Editor

Journal Requirements:

When submitting your revision, we need you to address these additional requirements.

1. Please ensure that your manuscript meets PLOS ONE's style requirements, including those for file naming. The PLOS ONE style templates can be found at 

https://journals.plos.org/plosone/s/file?id=wjVg/PLOSOne_formatting_sample_main_body.pdf and 

https://journals.plos.org/plosone/s/file?id=ba62/PLOSOne_formatting_sample_title_authors_affiliations.pdf

2. Please note that PLOS One has specific guidelines on code sharing for submissions in which author-generated code underpins the findings in the manuscript. In these cases, we expect all author-generated code to be made available without restrictions upon publication of the work. Please review our guidelines at https://journals.plos.org/plosone/s/materials-and-software-sharing#loc-sharing-code and ensure that your code is shared in a way that follows best practice and facilitates reproducibility and reuse.

3. Thank you for uploading your study's underlying data set. Unfortunately, the repository you have noted in your Data Availability statement does not qualify as an acceptable data repository according to PLOS's standards.

At this time, please upload the minimal data set necessary to replicate your study's findings to a stable, public repository (such as figshare or Dryad) and provide us with the relevant URLs, DOIs, or accession numbers that may be used to access these data. For a list of recommended repositories and additional information on PLOS standards for data deposition, please see https://journals.plos.org/plosone/s/recommended-repositories.

4. Please ensure that you refer to Figure 1 in your text as, if accepted, production will need this reference to link the reader to the figure.

5. We notice that your supplementary figures are uploaded with the file type 'Figure'. Please amend the file type to 'Supporting Information'. Please ensure that each Supporting Information file has a legend listed in the manuscript after the references list.​

6. If the reviewer comments include a recommendation to cite specific previously published works, please review and evaluate these publications to determine whether they are relevant and should be cited. There is no requirement to cite these works unless the editor has indicated otherwise. 

Additional Editor Comments:

Dear authors,

Thank you for submitting your paper. We have received the reviews on your manuscript. The reviewers have identified several areas of concern; however, they also see potential in your work. Therefore, I am pleased to invite you to submit a major revision of your manuscript.

Please carefully address all the comments and concerns raised by the reviewers in your revised version. You should also provide a point-by-point response letter explaining how you have addressed each comment.

We look forward to receiving your revised manuscript. Please note that the revised version will be sent back to the reviewers for further evaluation.

Best wishes.
Editor


# Reviewer 1

1. The permutation test in Appendix S3 needs one more look. If the authors only shuffle the order of the step vectors \Delta_t, their mean and covariance do not change, so the reported \Lambda should also stay unchanged. This does not quite add up with the reported p-values; please clarify the actual permutation procedure and correct the text or analysis if needed. 

2. Please clarify whether the MiniLM embeddings are L2-normalized before k-means, centroid-distance calculation, and drift analysis. Euclidean distance on unnormalized sentence embeddings can be affected by vector norm, so this small detail matters here. The related work on semantic representation could be slightly broadened. Studies such as ROUGE-SEM: Better evaluation of summarization using ROUGE combined with semantics, From coarse to fine: Enhancing multi-document summarization with multi-granularity relationship-based extractor, and Towards Curriculum Learning of Multi-Document Summarization Using Difficulty-Aware Mixture-of-Experts may provide some broader perspectives on semantic-aware evaluation and representation learning. These works are not direct topic-drift baselines, of course, but may be useful as methodological background. 

3. The fixed choice of k=50 for all 204 monthly windows is reasonable for scalability, but the corpus size changes dramatically over time. A short sensitivity discussion would help, especially for the very early low-volume years. 

4. Some conclusions are a bit too strong. Phrases such as “reflects durable shifts in meaning” or “evidence for polarization” go beyond what embedding-space movement alone can prove. I would tone these claims down a little.

5. The reproducibility part could be cleaner. Please provide the analysis code, main hyperparameters/seeds, and preferably the derived monthly centroids/topic trajectories, since reproducing the full 12.7-billion-comment pipeline is obviously not a piece of cake. 

6. A few figures are rather hard to read, especially the dense UMAP labels and heatmaps. Please enlarge the labels/legends and make clear again that the 2-D UMAP plots are only visualization, while the reported drift statistics are calculated in the original embedding space.

Overall, I like the basic idea and the large-scale analysis is interesting. If the permutation-test point is only a description issue, the remaining comments are mostly minor and easy to fix.


# Reviewer 2

The manuscript addresses an interesting problem and the large-scale Reddit analysis is impressive. However, several methodological issues should be clarified.

1. The whole analysis relies on all-MiniLM-L6-v2. It would be useful to repeat part of the experiment with another sentence-embedding model to show that the observed drift is not model-specific. 

2. A fixed k = 50 is used for every month, although the monthly corpus size changes dramatically over time. The authors should provide a stronger sensitivity analysis for different values of k. 

3. The corpus size in recent years is more than 160 times that of the early period. This imbalance may affect centroid stability and measured drift. A controlled subsampling analysis would be helpful. 

4. Topic alignment depends on UMAP followed by HDBSCAN. Since both steps may influence the resulting trajectories, sensitivity to UMAP/HDBSCAN parameters and random seeds should be reported. 

5. The random-walk test assumes that topic-step vectors are i.i.d. Gaussian. This is a strong assumption for temporally evolving discourse and deserves further justification or diagnostic testing. 

6. The analysis produces more than 1,000 aligned topic groups and reports significance using p<0.05. A multiple-testing correction such as FDR should be considered before identifying significantly drifting topics. 

7. The related work on temporal semantic representation could be broadened a little. Some studies such as An efficient loss function and deep learning approach for ranking stock returns in the absence of prior knowledge, A hierarchical deep model integrating economic facts for stock movement prediction, and Separating the predictable part of returns with CNN-GRU-attention from inputs to predict stock returns, may provide some broader methodological perspectives. 

8. The current clustering robustness analysis is mainly based on one selected time window. Testing several early, middle, and late periods would provide stronger evidence of stability. 

9. Some interpretations are stronger than the analysis supports. Changes in embedding geometry indicate changes in Reddit discourse representations, but statements about cultural realignment, polarization, or durable shifts in meaning should be phrased more cautiously. 

10. The Ethics Statement is currently listed as N/A. Given that the study analyzes billions of user-generated Reddit comments, a short discussion of privacy, public-data use, deleted users, and responsible handling of sensitive content would be appropriate.