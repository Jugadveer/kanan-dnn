# Early warning after the first term

A deep neural network that estimates, right after the first term, which of three
groups a student's final Portuguese grade will fall into: **Needs support** (0-9),
**On track** (10-13) or **High achiever** (14-20). At that point the school knows
the first-term grade G1 and the student's background; the second-term grade G2
arrives too late to act on, and the final grade G3 is what we predict.

The network is held to an honest standard: it is compared with the naive rule
"the final group equals the group of the first-term grade", which is already
right for 75 % of students, with confidence intervals and significance tests.

`REPORT.md` is the full project report; `VIVA_PREP.md` is the study guide.

## Headline results

On 98 test students the model never saw (95 % bootstrap intervals in brackets):

| Question | Neural network | Naive rule: group of G1 |
|---|---|---|
| Needs support? Students found (recall) | **1.000** (15 of 15) | 1.000 (15 of 15) |
| Needs support? Ranking by risk (ROC-AUC) | **0.974** (0.94-0.99) | 0.958 (0.93-0.99) |
| Which group? Ranking by probability (ROC-AUC) | **0.946** (0.91-0.98) | 0.906 (0.85-0.95) |
| Which group? Balanced accuracy | 0.879 (0.82-0.93) | 0.887 (0.82-0.94) |
| Which group? Macro F1 | 0.825 (0.74-0.90) | 0.850 (0.76-0.92) |

What this means: **the network finds every student who will need support, and
it ranks students by risk clearly better than the naive rule**: in repeated
cross-validation (15 folds) its ROC-AUC is 0.886 against 0.801, and 0.940
against 0.863 for the Needs-support question, better in **15 of 15 folds**.
**On the hard label it does not beat the rule**: balanced accuracy ties (0.756
against 0.752, ahead in 10 of 15 folds, p = 0.64; McNemar p = 0.45 on the test
set) and macro F1 is lower (0.702 against 0.721, behind in 12 of 15 folds).
After the first term, the first-term grade carries nearly all the information
in this data; the network's contribution is calibrated probabilities that put
students in order of risk, which is what a teacher deciding whom to see first needs.

## The brief, and where each part lives

| The project brief asks for | Where it is done |
|---|---|
| Data preprocessing | `earlywarning/loading.py` (groups, 70/15/15 split), `earlywarning/encoding.py` (embedding codes, scaling, one-hot) |
| Exploratory data analysis | `earlywarning/explore.py`; charts in `reports/charts/eda_*.png` |
| Feature selection | `earlywarning/selection.py`: RFECV with Logistic Regression and the one-standard-error rule, refitted inside every CV fold |
| DNN architecture design | `earlywarning/model.py` `build_network`: Keras functional API, entity embeddings, BatchNorm, dropout, L2 |
| Model training | `earlywarning/model.py` `NetworkClassifier`: α-balanced focal loss, early stopping, seed ensemble, temperature scaling |
| Hyperparameter tuning | `earlywarning/tuning.py` (Hyperband proposes) and `earlywarning/train.py` `choose_network` (cross-validation decides) |
| Performance evaluation | `earlywarning/evaluate.py`: three granularities, bootstrap intervals, McNemar, paired folds; `reports/results.json` |
| Comparison with traditional ML | `earlywarning/baselines.py`: KNN and decision tree (grid-searched by the same CV), naive G1 rule, two trivial models |
| Further analyses | `earlywarning/experiments.py`: repeated CV, ablation, learning curve, calibration, fairness, error analysis, cost of leaving out sex |
| Streamlit app | `app/`: Board, Check a student, Model report, What drives predictions, Experiments |
| Tests | `tests/`: 24 pytest tests, about 20 seconds |

## Dataset

[UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance),
Portuguese-language course (`student-por.csv`), downloaded by the code on first use.

| Check | Brief | This data |
|---|---|---|
| Students | 649 | 649 |
| Columns | about 30 attributes + G3 | 33: 30 attributes, G1, G2, G3 |
| Missing values / duplicate rows | | 0 / 0 |
| Inputs offered to the model | | 30: the 30 attributes and G1, minus sex |

| Group | Needs support | On track | High achiever |
|---|---|---|---|
| Final grade G3 | 0-9 | 10-13 | 14-20 |
| Students | 100 (15 %) | 355 (55 %) | 194 (30 %) |

**Why G1 in, G2 out.** The warning is given after the first term, when G1 is
known. G2 comes at the end of the second term, too late to change much.
**Why sex out.** An earlier version of this model leaned on sex more than on any
other input. Judging a student partly by sex is unfair to the individual, so sex
is never offered to the model; the experiments measure what that costs.

**What G1 already tells.** The group of G1 equals the final group for 488 of
649 students (75.2 %). Students tend to improve: on average G3 is 0.51 points
above G1, and 69 of the 157 students whose G1 was below 10 still finished On
track (`python -m earlywarning explore`).

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate                    # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m earlywarning explore            # data checks and EDA charts (seconds)
python -m earlywarning train              # tune, choose, train, test once (34 to 51 minutes measured, CPU shared with other jobs)
python -m earlywarning experiments        # repeated CV, ablation, ... (72 min measured)
python -m pytest                          # 24 tests (about 20 seconds)
python -m streamlit run app/Home.py       # the web app
```

Run everything from the project folder. Settings live in `config.yaml`; the
wording of every input lives in `columns.yaml`. Times were measured on a
laptop CPU shared with other jobs. Two training runs give identical results
(fixed seeds, deterministic TensorFlow operations). TensorFlow prints harmless
start-up messages (oneDNN, GPU support, `NodeDef`), and Hyperband prints
"Skipping variable loading for optimizer" when it resumes a candidate.

## Project structure

```
config.yaml           every setting: inputs left out, split, selection rule, search spaces, ...
columns.yaml          each column's group, question and value labels for the app
earlywarning/
  __main__.py         command line: python -m earlywarning explore | train | experiments
  settings.py         reads the YAML files and the theme once
  loading.py          download, read, grade -> group, 70/15/15 split, train+validation pool
  explore.py          EDA: checks, G1 vs G3, Spearman correlation, group mix charts
  encoding.py         flat one-hot table and the network's embedding inputs
  selection.py        RFECV + one-standard-error rule
  validation.py       leakage-free cross-validation (RFECV inside every fold)
  model.py            the functional network, focal loss, NetworkClassifier (ensemble, temperature)
  tuning.py           KerasTuner Hyperband
  baselines.py        naive G1 rule, trivial models, KNN, decision tree
  evaluate.py         metrics at three granularities, bootstrap intervals, McNemar, paired folds
  calibration.py      temperature scaling, Brier score, expected calibration error
  explain.py          permutation importance, per-student "what pushed this prediction"
  experiments.py      the long analyses
  charts.py           saved charts in the app's theme
  form.py             describes the app's input form
  predictor.py        loads the saved networks for the app
  train.py            the training pipeline, step by step
app/                  Home.py (header and top navigation), shared.py, pages/ (5 pages: Board,
                      Check a student, Model report, What drives predictions, Experiments)
.streamlit/           config.toml (colours, Barlow typeface), board.css (lamps, meters, watch list)
tests/                pytest suite
artifacts/            network_*.keras, encoder.joblib, serving.json, form.json (made by train)
reports/              results.json, experiments.json, importance.json, charts/
```

## Method in brief

1. **Split** 70 / 15 / 15 (453 / 98 / 98), stratified. Training + validation (551)
   form the pool for model selection; the test set is scored once at the end.
2. **Feature selection.** RFECV ranks one-hot columns with a class-balanced
   Logistic Regression and scores every column count by 5-fold CV. The
   **one-standard-error rule** (Breiman et al., 1984) then keeps the fewest
   columns that score within one standard error of the best. Run inside every
   CV fold, so no CV score has seen its held-out students during selection.
3. **Network.** Keras functional API: one embedding per text column plus
   standardised numbers, concatenated; dense layers with batch normalisation,
   ReLU, dropout and L2; softmax over 3 groups. α-balanced focal loss
   (γ = 2, α from group counts), Adam, early stopping (patience 25).
4. **Model selection.** Hyperband proposes (90 candidates, scored on the
   validation split); the best 3 distinct settings are re-scored by 5-fold CV
   on the 551 pool students; the same CV decides whether a 5-seed ensemble
   helps. KNN and the decision tree are grid-searched on the same folds.
5. **Calibration.** One temperature fitted on the validation students.
6. **Test once**: every model, three granularities, 95 % bootstrap intervals
   (2000 resamples), McNemar's exact test and paired CV folds against the naive
   rule and the best traditional model.

## Results

### Model selection (5-fold CV on the 551 training + validation students)

Feature selection: RFECV's best score came with 2 encoded columns; the
one-standard-error rule kept **1: the first-term grade G1** (on the training
split and in 4 of 5 CV folds; one fold also kept age, Fjob, failures and
higher). With RFECV's best count instead, the network scored 0.717.

| Hyperband candidate (hidden units, dropout, L2, learning rate) | Hyperband (98 validation students) | CV (551 students) |
|---|---|---|
| **64-128-16, 0.4, 0.01, 0.001** | 0.813 | **0.753 ± 0.033** |
| 128-16-32, 0.3, 0.0001, 0.0003 | 0.813 | 0.752 ± 0.034 |
| 64-32-64, 0.5, 0.0001, 0.001 | 0.813 | 0.743 ± 0.040 |

Hyperband could not tell its top candidates apart (all 0.813 on 98 students);
cross-validation could, a little. A 5-seed ensemble scored 0.750 against 0.753
for one network, so it is **not** used. Temperature on validation: T = 0.52
(the focal-loss network is under-confident; scaling sharpens it).

| Model (CV, mean of 5 folds) | Balanced accuracy | Macro F1 | ROC-AUC | Needs-support recall / precision |
|---|---|---|---|---|
| Neural network | **0.753** ± 0.033 | 0.701 | **0.883** | 0.847 / 0.537 |
| Group of first-term grade | 0.752 ± 0.031 | 0.721 | 0.801 | 0.859 / 0.544 |
| Decision tree (depth 2) | 0.747 ± 0.037 | **0.759** | 0.856 | 0.718 / 0.734 |
| KNN (k = 9, distance) | 0.728 ± 0.050 | 0.744 | 0.845 | 0.671 / 0.736 |
| Always On track | 0.333 | 0.236 | 0.500 | 0 / 0 |
| Random (stratified) | 0.315 | 0.313 | 0.486 | 0.129 / 0.144 |

### Test set (98 students, used once)

| Model | Balanced accuracy | Macro F1 | ROC-AUC | Needs-support recall | Needs-support precision |
|---|---|---|---|---|---|
| Neural network | 0.879 (0.82-0.93) | 0.825 (0.74-0.90) | **0.946** (0.91-0.98) | **1.000** | 0.682 (0.48-0.88) |
| Group of first-term grade | **0.887** (0.82-0.94) | **0.850** (0.76-0.92) | 0.906 (0.85-0.95) | **1.000** | 0.682 (0.48-0.88) |
| Decision tree | 0.817 (0.72-0.90) | 0.824 (0.73-0.90) | 0.903 (0.86-0.95) | 0.733 (0.50-0.94) | **0.733** (0.50-0.94) |
| KNN | 0.817 (0.72-0.90) | 0.824 (0.73-0.90) | 0.929 (0.88-0.97) | 0.733 (0.50-0.94) | **0.733** (0.50-0.94) |
| Always On track | 0.333 | 0.237 | 0.500 | 0 | 0 |
| Random (stratified) | 0.274 (0.19-0.36) | 0.273 (0.19-0.35) | 0.459 | 0.067 | 0.071 |

The test set is easier than average: the naive rule scores 0.887 here but 0.752
in cross-validation. KNN and the tree score identically on every label metric
(both effectively learn cut-offs on G1); only their probabilities, and so their
ROC-AUC, differ. On the test set within-one accuracy is 1.000 for every model except random
guessing: with three ordered groups only a Needs support ↔ High achiever mix-up
counts, and no trained model makes one, so it is no achievement.

### Is the network better? (significance)

| Network against | Test: only network right / only rival right | McNemar p | CV balanced accuracy difference (folds won) | CV ROC-AUC difference (folds won) |
|---|---|---|---|---|
| Naive rule | 2 / 5 | 0.453 | +0.001 (4 of 5) | **+0.082 (5 of 5)** |
| Decision tree | 6 / 8 | 0.791 | +0.006 (2 of 5) | **+0.027 (5 of 5)** |

On labels: no significant difference from either (and the naive rule has the
higher macro F1 in 4 of 5 folds). On ranking: the network is better in every
fold (paired t-test p = 0.0005 against the naive rule, 0.014 against the tree;
optimistic because folds overlap, Dietterich 1998, but 5 of 5 is consistent).

## Experiments (`python -m earlywarning experiments`, 71.9 minutes measured)

### Repeated cross-validation (5 folds × 3 repeats = 15 folds, RFECV inside each)

| Model | Balanced accuracy | Macro F1 | ROC-AUC | Needs-support ROC-AUC | Needs-support recall / precision |
|---|---|---|---|---|---|
| Neural network | **0.756** ± 0.048 | 0.702 ± 0.046 | **0.886** ± 0.022 | **0.940** ± 0.019 | **0.847** / 0.540 |
| Group of first-term grade | 0.752 ± 0.035 | 0.721 ± 0.034 | 0.801 ± 0.027 | 0.863 ± 0.040 | **0.859** / 0.546 |
| Decision tree | 0.746 ± 0.038 | **0.756** ± 0.045 | 0.857 ± 0.025 | 0.923 ± 0.026 | 0.714 / **0.736** |

| Network minus rival, over 15 folds | vs naive rule | vs decision tree |
|---|---|---|
| ROC-AUC | **+0.085, ahead in 15** | **+0.029, ahead in 15** |
| Needs-support ROC-AUC | **+0.078, ahead in 15** | **+0.018, ahead in 15** |
| Balanced accuracy | +0.003 (10 ahead, 5 behind; p = 0.64) | +0.010 (8 ahead, 6 behind; p = 0.36) |
| Macro F1 | −0.019 (3 ahead, 12 behind; p = 0.009) | −0.054 (1 ahead, 13 behind; p = 0.0003) |
| Needs-support recall | −0.012 (0 ahead, 2 behind, 13 tied) | **+0.133 (13 ahead, 2 tied)** |
| Needs-support precision | −0.006 (0 ahead, 2 behind, 13 tied) | −0.196 (13 behind, 2 tied) |

p: paired t-test over folds, optimistic because folds share students (Dietterich, 1998).
Reading: the network is significantly better at ranking; on hard labels it
matches the naive rule on balanced accuracy and loses on macro F1, because its
class weights trade precision for recall on the rare groups. Against the tree it
catches far more students who need support, at the cost of more false alarms.

### Ablation (5-fold CV, one network; one change at a time)

| Change | Balanced accuracy | Difference | Folds better / worse | Takeaway |
|---|---|---|---|---|
| Full design (reference) | 0.753 ± 0.033 | | | |
| Without G1 (no grades at all) | 0.556 ± 0.020 | **−0.197** | 0 / 5 | The first-term grade is the model. |
| No feature selection (all 30 inputs) | 0.718 ± 0.025 | −0.035 | 0 / 5 | Extra inputs are noise to the network. |
| No class weights (α = 1) | 0.728 ± 0.049 | −0.025 | 2 / 3 | Weighting rare groups matters for balanced accuracy. |
| RFECV's best count, not the 1-SE rule | 0.736 ± 0.019 | −0.017 | 1 / 3 | The simpler choice generalises better. |
| Seed ensemble of 5 | 0.750 ± 0.036 | −0.003 | 0 / 1 | No gain, so not used. |
| Cross-entropy, not focal (γ = 0) | 0.750 ± 0.035 | −0.003 | 0 / 1 | Focal loss barely matters here. |
| No L2 | 0.750 ± 0.035 | −0.003 | 0 / 1 | Within noise. |
| One-hot instead of embeddings | 0.754 ± 0.032 | +0.001 | 1 / 0 | No text inputs survive selection in 4 of 5 folds. |
| No batch normalisation | 0.758 ± 0.030 | +0.005 | 1 / 0 | Within noise; kept as designed. |
| Sex added back as an input | 0.753 ± 0.033 | 0.000 | 0 / 0 | RFECV never selects sex: leaving it out costs nothing. |

Folds better / worse: folds where the change scored above / below the reference
(ties not shown). Differences under about 0.01 are well inside the ± 0.03 fold
spread and should be read as "no effect".

### Learning curve (5-fold CV, RFECV redone on every reduced training set)

| Training students per fold | 88 | 176 | 264 | 352 | 440 |
|---|---|---|---|---|---|
| Neural network | 0.678 | 0.751 | 0.765 | 0.756 | 0.753 |
| Decision tree | 0.737 | 0.697 | 0.724 | 0.731 | 0.747 |
| Group of first-term grade | 0.752 | 0.752 | 0.752 | 0.752 | 0.752 |

The network needs about 176 students to reach the naive rule and is flat after
that: more students of the same kind would not help much; new information would.

### Calibration (test set, 5 confidence bins)

| | Expected calibration error | Brier score | Log loss |
|---|---|---|---|
| Before temperature scaling | 0.220 | 0.372 | 0.583 |
| After (T = 0.52, fitted on validation) | **0.140** | **0.325** | **0.498** |

The raw network was under-confident (in its lowest-confidence bin it was right 81 %
of the time while claiming 52 %); scaling sharpens it. The lowest bin is still
under-confident after scaling (claims 57 %, right 97 %), so its "unclear" cases
are often right. The app's wording follows the calibrated probability.

### Fairness audit (test set; sex is not an input)

| Group | Students | Needs support (actual / flagged) | Network accuracy | Naive accuracy | Needs-support recall |
|---|---|---|---|---|---|
| Female | 52 | 6 / 9 | 0.85 | 0.87 | 1.00 |
| Male | 46 | 9 / 13 | 0.80 | 0.85 | 1.00 |
| School GP | 63 | 4 / 8 | 0.81 | 0.83 | 1.00 |
| School MS | 35 | 11 / 14 | 0.86 | 0.91 | 1.00 |
| Rural | 32 | 7 / 10 | 0.81 | 0.88 | 1.00 |
| Urban | 66 | 8 / 12 | 0.83 | 0.85 | 1.00 |

Every group's struggling students are all found; every group gets some false
alarms. Boys are flagged more often (28 % against 17 %) because more of them
truly need support (20 % against 12 %), not because of their sex: the inputs
(G1) predict sex with ROC-AUC 0.555, close to chance. Groups of 32-66 students:
accuracy gaps of 0.05 are within chance.

### Error analysis (test set)

17 of 98 students are in the wrong group; none is two groups off. In 12 of the 17
the network predicted the group of the first-term grade. The mistaken students
moved +1.0 points from G1 to G3 on average, against +0.43 overall. The most
confident mistakes are students near a boundary who crossed it: G1 = 8 who
finished on 10-11 (predicted Needs support, were On track), G1 = 14 who slipped to
12-13, G1 = 11 who rose to 14. With G1 as the only input, these students look
identical to those who did not move; nothing in the data separates them.

### Cost of leaving sex out

Trained with sex allowed, RFECV again kept only G1, so the model and its test
scores are identical (balanced accuracy 0.879, Needs-support recall 1.000). In
this framing the fairness choice costs nothing. (In the earlier start-of-year
version, without G1, the network had relied on sex more than on any other input.)

## Limitations

- **The network does not beat the first-term rule on the label.** Every
  principled change tried was measured by leakage-free CV: plain cross-entropy
  (0.717), no class weights (0.667), a "group of G1" feature (0.706-0.712), a
  smaller network (0.725), all 30 inputs (0.705), RFECV's best count (0.717);
  the one-standard-error rule (0.753) is the best and is used. These probes used
  the same 5 folds, so choosing among them is itself a small selection effect.
- **Small data.** 551 students for model selection, 98 for testing: every score
  moves by a few points with a different sample; read the intervals.
- **The test set is easier than average** (naive rule 0.887 there, 0.752 in CV),
  so test scores flatter every model; the CV numbers are the realistic ones.
- **One school system, one subject, one cohort** (Portuguese course, two
  schools in Portugal, 2008 study). Nothing here shows the model transfers.
- **Ordered groups treated as unordered classes**; an ordinal loss is a natural next step.
- **Fairness.** Sex is not an input, but school and family background can act as
  proxies when selected; with G1 alone the proxy risk is low but G1 itself
  reflects earlier inequality. The model should prompt a teacher, never decide.
