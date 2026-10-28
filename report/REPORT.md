# Selecting Reliable LLM Answers with a Small Pairwise Ranker

DASE7506 Mini-project 2

## 1. Task and research question

This project selects one of four supplied solutions to a mathematics question. The official evaluator checks the selected solution's parsed final number, not the validity of its reasoning. All groups contain at least one correct candidate, and several candidates may be correct. The method must run offline on CPU with four total threads, at most 16 MiB of extra inference assets, and at most 30 seconds for fresh prediction plus independent scoring on the reference Xeon Platinum 8457C.

The main question is whether a small linear model of the supplied whole-solution representations improves selection when candidate answers disagree, and whether agreement between solutions adds useful evidence. A secondary question is whether gains can be explained by simple length, formatting and answer-value features. I expected within-question learning and agreement to help; the experiments support the first expectation more clearly than the second.

## 2. Data and protocol

The supplied splits contain 4,755 training, 798 validation and 467 test questions, each with four candidates and a 1,024-dimensional representation. Only training inputs and references are used to fit parameters or transformations. Validation is used for design and selection. Test results are obtained only after freezing the method. No external dataset, extra pretrained weights, LLM API, answer lookup or cached prediction is used in inference.

| Property | Training | Validation |
|---|---:|---:|
| Questions | 4,755 | 798 |
| All four candidates correct | 2,242 | 379 |
| At least two distinct valid answers | 2,114 | 355 |
| Parseable candidates / all candidates | 18,149 / 19,020 | 3,035 / 3,192 |
| Mean characters, correct / wrong candidates | 464.84 / 527.41 | 456.42 / 548.33 |

Nearly half of training and validation groups have four correct answers. The official guide states that test groups have at least two distinct valid parsed answers. Consequently, I report both full validation accuracy and accuracy on the 355-question disagreement subset. This subset is defined from candidate text alone, without correctness labels. Full validation accuracy is an optimistic description of performance on the harder test construction.

Labels compare each candidate and reference using the unchanged official parser. Multiple matching candidates all receive label 1. Unparseable candidates receive 0. Question IDs and text are checked for alignment but are not features. Final numeric correctness does not imply sound reasoning.

## 3. Method

The main inference interface is `selection.predict(bank, arrays, checkpoint)`. It returns native Python integers in bank order and uses only the supplied inputs and a small configuration/weight bundle. All methods except the deliberately simple first-candidate baseline mask unparseable answers. Ties are resolved by the earliest candidate; an all-invalid external group falls back to index 0.

### Features

The selected method concatenates the supplied 1,024-dimensional embedding and 12 surface/answer features: log(1+character count), word count, newline count, digit-run count, equals count, backslash count and bold-marker count; presence of #### and boxed markers; parseability; signed log(1+absolute answer), with magnitude capped at 10^12; and integer-answer status. The log transform applies to all seven counts. No new language-model representation is generated.

Each feature is standardized using only parseable training candidates. Constant dimensions use scale 1. A separate full-feature family adds three answer-agreement features: frequency among four candidates, membership in a maximum-frequency group, and whether the answer occurs once. These features are computed independently for each new question.

### Pairwise training

For question q, let z_qi be its standardized candidate vector and P_q the set of ordered correct/incorrect pairs for which both answers parse. For each pair (i,j), the ranker learns w^T(z_qi-z_qj) approximately equal to 1. The objective is:

L(w) = (1/Q) sum_q [(1/|P_q|) sum_(i,j in P_q) (w^T(z_qi-z_qj)-1)^2] + alpha ||w||^2.

Q counts questions with at least one such pair. Equal question weighting prevents groups with more pairs from dominating. Groups with all valid answers correct do not produce pairs. A closed-form ridge solve gives deterministic weights. At inference, the method selects the parseable candidate with maximum linear score. No per-question retraining occurs.

The pointwise comparison instead regresses independent valid-candidate correctness labels with a fitted constant mean. Comparing pointwise and pairwise models therefore changes the objective, effective training examples and question weighting together; it is not an isolated test of one mathematical term.

### Agreement and search

For each learned ranker, I compare direct candidate scoring with averaging scores across candidates that share the same canonical numeric answer. This pooling can reduce individual noise but can also average away evidence from the strongest candidate.

The predeclared grid has five feature/objective families, alpha in {0.1, 1, 10}, and two inference pooling choices: 15 fits, 30 learned configurations, plus first-candidate, majority and shortest-valid baselines. Fitting is deterministic; seed 20261024 is recorded. Models are selected by disagreement-subset correct count, then full validation correct count, then smaller assets, then experiment ID. Validation is reused for selection, so it is not an independent estimate of generalization.

## 4. Validation results and controlled comparisons

All 33 configurations were evaluated through the unchanged official runner and independent scorer. The complete search took 36.80 seconds locally, including data preparation, fits and repeated validation evaluation. This is search cost, not a single inference measurement. The table holds alpha=0.1 for learned-model comparisons; remaining regularization results are retained in the experiment JSON.

| Method | Full correct / 798 | Full % | Disagreement correct / 355 | Disagreement % |
|---|---:|---:|---:|---:|
| First candidate | 624 | 78.20 | 195 | 54.93 |
| Majority vote | 688 | 86.22 | 245 | 69.01 |
| Shortest valid | 584 | 73.18 | 141 | 39.72 |
| Surface, pairwise | 701 | 87.84 | 258 | 72.68 |
| Embedding, pointwise | 726 | 90.98 | 283 | 79.72 |
| Embedding, pairwise | 735 | 92.11 | 292 | 82.25 |
| Embedding + surface, pairwise (selected) | 739 | 92.61 | 296 | 83.38 |
| Full features including agreement | 733 | 91.85 | 290 | 81.69 |
| Selected features + answer-mean pooling | 729 | 91.35 | 286 | 80.56 |

The selected ranker improves disagreement accuracy by 14.37 percentage points over majority vote, with 74 corrections and 23 regressions on the same 355 questions. Its net improvement is 51 questions. Surface features alone exceed majority by 13 questions but remain 38 questions behind the selected model. Choosing the shortest valid text performs poorly despite wrong candidates being longer on average; the aggregate length correlation is insufficient for reliable individual decisions.

With embeddings only and the same alpha, pairwise training improves over pointwise by 9 questions (25 corrections, 16 regressions). Adding surface features to the pairwise embedding model yields only 4 net questions (11 corrections, 7 regressions). These smaller differences should not be overstated on a selection set of 355 questions.

The agreement hypothesis is not consistently supported. Adding agreement features to the selected feature family loses 6 questions at alpha=0.1; averaging the selected ranker's scores within answers loses 10 questions. With agreement features already present, pooling sometimes helps, but the best full-feature pooled configuration still falls short of the selected unpooled method. More structure is not automatically better.

Exploratory paired McNemar calculations are saved with all comparisons. The selected-versus-majority comparison has a small descriptive p-value, whereas the pairwise-versus-pointwise and surface-addition comparisons do not provide strong evidence individually. These tests are post-selection and not corrected for multiple comparisons; they are not confirmatory significance claims. The unchanged final test is the separate evaluation.

## 5. What errors reveal

Examples were selected reproducibly: the first two validation questions, in bank order, for each of three categories (improvement over majority, regression, both wrong). They were not hand-picked to maximize favorable examples. Full candidate texts and indices are in `records/analysis/validation-error-examples.json`.

### Better final answers can still contain bad reasoning

For a concert lasting 80 minutes with a 10-minute intermission and one double-length song, the selected candidate reaches 13 songs, while vote tie-breaking chooses 16. However, that selected explanation invents incompatible start/end times before using the correct duration. It is correct under final-answer grading but not uniformly reliable reasoning.

For a plant producing two baby plants twice yearly for four years, the selected candidate gives the correct number 16. Its explanation incorrectly describes a doubling sequence instead of the intended multiplication 2 x 2 x 4. This is a particularly clear example of a numerically correct answer reached through faulty intermediate statements. High selection accuracy must not be presented as proof of step-by-step verification.

### Regressions on interpretable arithmetic

For 24 eggs, with 3 broken and 6 cracked, the selected answer is 6 instead of 9. The text first computes 15 perfect eggs, then incorrectly includes broken eggs in the cracked count. For four pies after selling one and giving away one, another selected candidate incorrectly claims that three pies remain and returns 6 pieces instead of 4. A linear representation score does not explicitly enforce these constraints.

### Benchmark semantics and ambiguity

For a paperclip problem, the supplied reference accepts 9, while three candidates and the selected model return 17. The wording can encourage interpreting "1/4 more than" as an additive increase, whereas the reference follows a different interpretation. The model selection is scored as wrong under the supplied reference, which is kept unchanged. It illustrates that reference disagreement and invalid reasoning are not interchangeable concepts.

On the square-dancing problem, the correct unmatched count is 2 but both majority and the ranker fail. Such cases show that neither agreement nor a learned plausibility score guarantees faithful use of every quantity.

### Limitations

The embeddings are fixed whole-solution representations prepared by the course. The experiment does not localize which tokens or reasoning steps contain useful evidence. Some predictive signal may be formatting, numeric magnitude or generation artifacts. The surface model is a partial control, not a proof that all shortcuts have been removed. The method does not produce calibrated probabilities or abstain. The validation disagreement subset improves relevance to the test construction but remains a development set, and no reference-machine performance measurement is available locally.

## 6. Resources, frozen evaluation and reproduction

The local CPU is an Intel Core i9-12900H running Windows 11. Python is 3.12.14, NumPy is 2.5.3, and threadpoolctl 3.6.0 verifies one loaded OpenBLAS pool with four threads. Execution is sequential. The reference Xeon Platinum 8457C has not been measured; eligibility ultimately depends on that machine.

The selected checkpoint contains config.json (146 bytes) and weights.npz. Counting the uncompressed NumPy archive members plus configuration gives 25,530 bytes, about 0.0244 MiB, well below 16 MiB. The weight archive is uncompressed; all learned means, scales, weights and bias are included. The observed validation prediction-plus-scoring time is about 0.89 seconds for 798 questions. A fresh training replay takes about 1.74 seconds including loading and feature preparation; both checkpoint files and all validation choices reproduce exactly. A separate clean virtual environment, installed offline from local wheels and using a Git export of the frozen source, reproduces all 467 test predictions and the byte-identical checkpoint after retraining. Prediction plus scoring in that environment takes 1.2876 seconds; all details are retained in records/reproduction/clean-environment.json.

**Frozen test result: 384/467 correct, accuracy 82.22698072805139%.** The method and assets were frozen at development commit cf8e1ca64c8d51ae0edb04e276acebbb38c56398 before evaluation. This standalone release preserves their hashes; its submitted Git revision identifies the published package. See records/release/source-mapping.json and README.md for the mapping; the development history is not required for reproduction. Full CPU prediction plus independent scoring took 0.7545 seconds locally, including fresh process startup. The test score is consistent with the validation disagreement estimate of 83.38%, rather than the easy-group-inflated full validation score of 92.61%. No algorithm or hyperparameter was changed after seeing this result. Official predictions, score JSON, commands, timing and freeze hashes are retained in records/final/.

The main environment is installed with `python -m pip install -r requirements.txt` under Python 3.12. Exact PowerShell commands are recorded in README.md. The prediction command uses `code/predict.py --method code/selection.py` from the project root and an explicit checkpoint path. Scoring uses `code/score_predictions.py` and the generated prediction file. An outer wall-clock timer includes both fresh processes, startup, imports, loading and preprocessing.

To reproduce fitting, `code/train_model.py` takes the saved configuration, a new output directory and a record path. It loads training data only and refuses to overwrite a model directory. To reproduce the search, run `code/run_experiments.py` in a fresh checkout; it refuses to overwrite the prior search directory. No retraining is needed for checkpoint evaluation.

Robustness checks passed for the full validation bank, a reversed 37-question subset, renamed IDs, a singleton and an empty bank. Prediction was tested from an empty working directory with an audit hook allowing only checkpoint reads and rejecting network access. It opened only config.json and weights.npz. These are input-contract checks, not a substitute for the instructor's audit bank.

## 7. Conclusion and disclosure

A small pairwise linear ranker is a strong validation alternative to majority voting. The main gain comes from the supplied representations; explicit agreement and answer pooling do not reliably improve the best model. Smaller ablation differences remain uncertain after model selection. Error examples show why answer accuracy and reasoning validity must be reported separately.

Codex assisted with requirements checking, environment setup, experimental design, coding, testing and report drafting. The student carried out the main experimental design, code review, test review and report refinement. Official tools and data were reused unchanged. Questions/references derive from GSM8K under the retained MIT license; candidate solutions and embeddings were prepared by the course with Qwen3-0.6B. No model weights were downloaded.

References: [1] DASE7506 MP2 Guide, github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md. [2] Official code README and data manifest in the same repository. [3] Retained source attribution and license in code/data/NOTICE.txt and LICENSE.txt. Full source links, commands and evidence are included in the accompanying repository.
