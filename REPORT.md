# Early Warning After the First Term: Predicting Students' Final Grade Group with a Deep Neural Network

## Abstract

Schools want to know which students will struggle while there is still time to
help. This project predicts, right after the first term, which of three groups
a student's final Portuguese grade (G3, 0–20) will fall into: Needs support
(0–9), On track (10–13) or High achiever (14–20). The data is the UCI Student
Performance dataset (Portuguese course, 649 students). The inputs are the
first-term grade G1 and 29 background details; the second-term grade G2 is
left out because it arrives too late to act on, and sex is left out for
fairness.

The model is a Keras functional network with entity embeddings, batch
normalisation, dropout, L2 and an α-balanced focal loss. Hyperband proposed
the settings and 5-fold cross-validation chose among them. Feature selection
(RFECV with the one-standard-error rule, refitted inside every fold) kept only
G1. The network is compared with KNN, a decision tree and the naive rule
"final group = group of G1", which is already right for 75 % of students.

On 98 held-out test students the network found all 15 students who needed
support (recall 1.000) and ranked students by risk better than the naive rule
(ROC-AUC 0.946 against 0.906). In 15-fold repeated cross-validation its ROC-AUC
was higher than the rule's in all 15 folds (0.886 against 0.801). On the hard
group label it did not beat the rule: balanced accuracy tied (0.756 against
0.752, p = 0.64) and macro F1 was lower (0.702 against 0.721). The conclusion is
that after the first term, G1 carries almost all the usable signal in this
data; what the network adds is calibrated probabilities that put students in
order of risk.

## 1. Introduction and objectives

A warning given at the end of the year is too late. A warning given after the
first term leaves two terms to arrange extra lessons, a mentor or an
attendance check. At that point a school knows the student's first-term grade
and background, and nothing else about the current year.

Objectives:

1. Predict the final group from what is known after the first term.
2. Find the students who will need support (recall on Needs support comes first).
3. Give probabilities a teacher can read at face value (calibration).
4. Compare the network honestly with traditional models and with the naive
   G1 rule, using confidence intervals and significance tests.
5. Check that the model treats groups of students fairly.
6. Deliver the model as a Streamlit app a head of year could use.

## 2. Dataset

UCI Student Performance (Cortez and Silva, 2008), file `student-por.csv`:
649 students from two Portuguese secondary schools, 33 columns (30 attributes,
G1, G2, G3), no missing values and no duplicate rows. The project groups the
attributes as the brief asks: Demographic, Academic, Social and Behavioural
(`columns.yaml`).

| Group | Needs support | On track | High achiever |
|---|---|---|---|
| Final grade G3 | 0–9 | 10–13 | 14–20 |
| Students | 100 (15 %) | 355 (55 %) | 194 (30 %) |

**Inputs offered to the model:** G1 and 29 attributes (30 attributes minus sex).
G2 is left out because it is known only at the end of the second term. Sex is
left out because an earlier, start-of-year version of this model relied on it
more than on any other input, and judging a student partly by sex is unfair to
the individual. Section 6.6 measures what that choice costs.

**Split:** stratified 70 / 15 / 15, giving 453 training, 98 validation and 98
test students (seed 2024). Training and validation together (551 students) form
the pool for cross-validated model selection. The test set is scored once, at
the end.

## 3. Exploratory analysis

`python -m earlywarning explore` produces the checks and the charts in
`reports/charts/eda_*.png`.

**G1 against the final group:**

| Group of G1 \ final group | Needs support | On track | High achiever |
|---|---|---|---|
| Needs support (G1 0–9) | 88 | 69 | 0 |
| On track (G1 10–13) | 12 | 267 | 61 |
| High achiever (G1 14–20) | 0 | 19 | 133 |

The diagonal holds 488 of 649 students (75.2 %): this is the naive rule's
accuracy and the bar any model must clear. Students tend to improve (G3 is on
average 0.51 points above G1), so the rule's main error is the 69 students with
G1 below 10 who still finished On track. No student moves two groups.

**Spearman rank correlation with G3:** G1 +0.88, failures −0.45, mother's
education +0.28, study time +0.27, father's education +0.23, weekday alcohol
−0.21. Every background detail is far weaker than G1, and most of what it
says about G3 is already in G1.

## 4. Methodology

### 4.1 Preprocessing

Two encodings, each fitted only on the rows a model trains on:

- **Flat** (KNN, decision tree, and the Logistic Regression inside RFECV):
  numbers standardised, text columns one-hot encoded.
- **Network** (`EmbeddingEncoder`): numbers standardised; each text column
  turned into integer codes, with 0 reserved for a value never seen in training.

### 4.2 Feature selection

RFECV (Guyon et al., 2002) fits a class-balanced Logistic Regression on the 55
encoded columns, drops the column with the smallest coefficients, and repeats,
scoring every column count by 5-fold balanced accuracy. The
**one-standard-error rule** (Breiman et al., 1984) then keeps the fewest
columns whose score is within one standard error of the best, because smaller
differences cannot be told apart from noise. Selection is refitted inside every
cross-validation fold, so no fold's score has seen its held-out students.

### 4.3 Network

Built with the Keras functional API (`build_network` in `earlywarning/model.py`):

```
text columns  -> one Embedding each --\
numbers (G1, ...) -> standardised -----+-> Concatenate
   -> [Dense (no bias) -> BatchNorm -> ReLU -> Dropout] x N -> Dense 3 softmax
```

- **Entity embeddings** (Guo and Berkhahn, 2016) give each value of a text
  column a small learnt vector. With only G1 selected, this branch is empty in
  the final model; it is used whenever selection keeps a text column.
- **Batch normalisation** (Ioffe and Szegedy, 2015) after each dense layer; the
  dense layers have no bias because batch normalisation adds its own shift.
- **Dropout** (Srivastava et al., 2014) and **L2** weight decay against
  overfitting.
- **α-balanced focal loss** (Lin et al., 2017): FL = −α_c (1 − p)^γ log p with
  γ = 2. α_c = n / (3 n_c) from the training labels: 2.157 for Needs support,
  0.611 for On track, 1.110 for High achiever. Rare groups count more, and
  students the network already gets right count less.
- **Adam** (Kingma and Ba, 2015), batch size 32, early stopping on validation
  loss with patience 25, best weights restored.

### 4.4 Model selection

1. **Hyperband** (Li et al., 2018) tried 90 settings (1–3 hidden layers, 16–128
   units, dropout 0.1–0.5, L2, learning rate, embedding size), scoring each on
   the 98 validation students.
2. Its three best distinct settings were re-scored by 5-fold cross-validation
   on the 551 pool students, which chose the final settings. Hyperband alone
   could not separate them: all three scored 0.813 on validation.
3. The same cross-validation decided whether a 5-seed ensemble helps. It did
   not, so the final model is one network.
4. KNN (k from 3 to 35, uniform or distance weights) and the decision tree
   (depth 2 to 8, minimum leaf 1 to 20) were grid-searched on the same folds.

### 4.5 Calibration

Temperature scaling (Guo et al., 2017): the network's log-probabilities are
divided by one temperature T, fitted on the validation students by minimising
log loss. The predicted group never changes; only the confidence does.

### 4.6 Evaluation

Each model is scored on three questions:

- **Which group?** Balanced accuracy (mean recall over the three groups), macro
  F1 and one-vs-rest ROC-AUC.
- **Needs support or not?** Recall, precision, ROC-AUC and accuracy.
- **At most one group off?** Within-one accuracy.

Test scores carry 95 % bootstrap intervals (2000 resamples; Efron and
Tibshirani, 1993). The network is compared with the naive rule and with the
best traditional model by McNemar's exact test on the test set (McNemar, 1947)
and by paired cross-validation folds. The paired t-test over folds is
optimistic because folds share training students (Dietterich, 1998), so the
number of folds won is reported beside it.

## 5. Results

### 5.1 Feature selection

On the training split RFECV scored best with 2 encoded columns (balanced
accuracy 0.772); a single column scored 0.769, well within one standard error.
The one-standard-error rule therefore kept **one column: G1**. In the five
cross-validation folds it kept G1 alone four times, and G1 with age, father's
job, failures and plans for higher education once. No background detail
improved the score once G1 was known; with more columns the score fell
steadily, to about 0.72 with all 55.

### 5.2 Final network

Input G1 → Dense 64 → Dense 128 → Dense 16 (each with batch normalisation,
ReLU, dropout 0.4, L2 0.01) → Dense 3 softmax; learning rate 0.001; 10,771
trainable weights; early stopping after 191 epochs. Temperature T = 0.52: the
focal loss leaves the network under-confident, so scaling sharpens it.

| Hyperband candidate (units, dropout, L2, learning rate) | Validation (98) | CV (551) |
|---|---|---|
| **64-128-16, 0.4, 0.01, 0.001** | 0.813 | **0.753 ± 0.033** |
| 128-16-32, 0.3, 0.0001, 0.0003 | 0.813 | 0.752 ± 0.034 |
| 64-32-64, 0.5, 0.0001, 0.001 | 0.813 | 0.743 ± 0.040 |

### 5.3 Cross-validation (5 folds, 551 students)

| Model | Balanced accuracy | Macro F1 | ROC-AUC | Needs-support recall / precision |
|---|---|---|---|---|
| Neural network | **0.753** ± 0.033 | 0.701 | **0.883** | 0.847 / 0.537 |
| Naive rule (group of G1) | 0.752 ± 0.031 | 0.721 | 0.801 | 0.859 / 0.544 |
| Decision tree (depth 2) | 0.747 ± 0.037 | **0.759** | 0.856 | 0.718 / 0.734 |
| KNN (k = 9, distance) | 0.728 ± 0.050 | 0.744 | 0.845 | 0.671 / 0.736 |
| Always On track | 0.333 | 0.236 | 0.500 | 0 / 0 |
| Random (stratified) | 0.315 | 0.313 | 0.486 | 0.129 / 0.144 |

### 5.4 Test set (98 students, used once)

| Model | Balanced accuracy | Macro F1 | ROC-AUC | Needs-support recall | Needs-support precision |
|---|---|---|---|---|---|
| Neural network | 0.879 (0.82–0.93) | 0.825 (0.74–0.90) | **0.946** (0.91–0.98) | **1.000** | 0.682 (0.48–0.88) |
| Naive rule (group of G1) | **0.887** (0.82–0.94) | **0.850** (0.76–0.92) | 0.906 (0.85–0.95) | **1.000** | 0.682 (0.48–0.88) |
| Decision tree | 0.817 (0.72–0.90) | 0.824 (0.73–0.90) | 0.903 (0.86–0.95) | 0.733 (0.50–0.94) | **0.733** (0.50–0.94) |
| KNN | 0.817 (0.72–0.90) | 0.824 (0.73–0.90) | 0.929 (0.88–0.97) | 0.733 (0.50–0.94) | **0.733** (0.50–0.94) |
| Always On track | 0.333 | 0.237 | 0.500 | 0 | 0 |
| Random (stratified) | 0.274 (0.19–0.36) | 0.273 (0.19–0.35) | 0.459 | 0.067 | 0.071 |

On the Needs-support question alone the network's ROC-AUC is 0.974 and its
accuracy 0.929. The test set is easier than average (the naive rule scores
0.887 here against 0.752 in cross-validation), so the cross-validation numbers
are the realistic ones. Within-one accuracy is 1.000 for every trained model:
with three ordered groups only a Needs support ↔ High achiever confusion
counts, and none occurs, so it is not reported as an achievement.

### 5.5 Is the network better?

| Network against | Test: only network right / only rival right | McNemar p | CV balanced accuracy difference (folds won) | CV ROC-AUC difference (folds won) |
|---|---|---|---|---|
| Naive rule | 2 / 5 | 0.453 | +0.001 (4 of 5) | **+0.082 (5 of 5)**, t-test p = 0.0005 |
| Decision tree | 6 / 8 | 0.791 | +0.006 (2 of 5) | **+0.027 (5 of 5)**, t-test p = 0.014 |

On the group label the network cannot be told apart from either rival. On
ranking students by probability it is better in every fold.

## 6. Experiments

`python -m earlywarning experiments` (71.9 minutes measured) writes
`reports/experiments.json` and the charts shown on the app's Experiments page.

### 6.1 Repeated cross-validation (5 folds × 3 repeats, RFECV inside each)

| Model | Balanced accuracy | Macro F1 | ROC-AUC | Needs-support ROC-AUC |
|---|---|---|---|---|
| Neural network | **0.756** ± 0.048 | 0.702 ± 0.046 | **0.886** ± 0.022 | **0.940** ± 0.019 |
| Naive rule | 0.752 ± 0.035 | 0.721 ± 0.034 | 0.801 ± 0.027 | 0.863 ± 0.040 |
| Decision tree | 0.746 ± 0.038 | **0.756** ± 0.045 | 0.857 ± 0.025 | 0.923 ± 0.026 |

| Network minus rival, over 15 folds | vs naive rule | vs decision tree |
|---|---|---|
| ROC-AUC | **+0.085, ahead in 15** | **+0.029, ahead in 15** |
| Needs-support ROC-AUC | **+0.078, ahead in 15** | **+0.018, ahead in 15** |
| Balanced accuracy | +0.003 (10 ahead, 5 behind; p = 0.64) | +0.010 (8 ahead, 6 behind; p = 0.36) |
| Macro F1 | −0.019 (3 ahead, 12 behind; p = 0.009) | −0.054 (1 ahead, 13 behind; p = 0.0003) |
| Needs-support recall | −0.012 (13 tied) | **+0.133 (13 ahead, 2 tied)** |
| Needs-support precision | −0.006 (13 tied) | −0.196 (13 behind, 2 tied) |

The network ranks better everywhere and ties on balanced accuracy. It loses on
macro F1 because its class weights buy recall on the rare groups at the cost of
precision. Against the tree it finds far more of the students who need support,
with more false alarms.

### 6.2 Ablation (5-fold CV, one change at a time; reference 0.753)

| Change | Balanced accuracy | Difference | Folds better / worse |
|---|---|---|---|
| Without G1 (no grades at all) | 0.556 | **−0.197** | 0 / 5 |
| No feature selection (all inputs) | 0.718 | −0.035 | 0 / 5 |
| No class weights (α = 1) | 0.728 | −0.025 | 2 / 3 |
| RFECV's best count instead of the 1-SE rule | 0.736 | −0.017 | 1 / 3 |
| Seed ensemble of 5 networks | 0.750 | −0.003 | 0 / 1 |
| Cross-entropy instead of focal loss (γ = 0) | 0.750 | −0.003 | 0 / 1 |
| No L2 | 0.750 | −0.003 | 0 / 1 |
| One-hot inputs instead of embeddings | 0.754 | +0.001 | 1 / 0 |
| No batch normalisation | 0.758 | +0.005 | 1 / 0 |
| Sex added back as an input | 0.753 | 0.000 | 0 / 0 |

The fold spread is about ± 0.03, so differences below about 0.01 mean no
effect. Two rows matter. Without G1 the network collapses, so G1 is the model.
Giving the network the background details back, through RFECV's best count or
through all inputs, lowers the score, and with all inputs it is lower in every
fold.

### 6.3 Learning curve

| Training students per fold | 88 | 176 | 264 | 352 | 440 |
|---|---|---|---|---|---|
| Neural network | 0.678 | 0.751 | 0.765 | 0.756 | 0.753 |
| Decision tree | 0.737 | 0.697 | 0.724 | 0.731 | 0.747 |
| Naive rule | 0.752 | 0.752 | 0.752 | 0.752 | 0.752 |

The network reaches the naive rule at about 176 students and then levels off.
More students of the same kind would help little; new information would.

### 6.4 Calibration (test set, 5 confidence bins)

| | Expected calibration error | Brier score | Log loss |
|---|---|---|---|
| Before temperature scaling | 0.220 | 0.372 | 0.583 |
| After (T = 0.52) | **0.140** | **0.325** | **0.498** |

Scaling cut the calibration error by a third. The lowest-confidence bin is
still under-confident (it claims 57 % and is right 97 % of the time), so the
app's "unclear" cases are more often right than the wording suggests.

### 6.5 Fairness audit (test set)

| Group | Students | Needs support (actual / flagged) | Network accuracy | Naive accuracy | Needs-support recall |
|---|---|---|---|---|---|
| Female | 52 | 6 / 9 | 0.85 | 0.87 | 1.00 |
| Male | 46 | 9 / 13 | 0.80 | 0.85 | 1.00 |
| School GP | 63 | 4 / 8 | 0.81 | 0.83 | 1.00 |
| School MS | 35 | 11 / 14 | 0.86 | 0.91 | 1.00 |
| Rural | 32 | 7 / 10 | 0.81 | 0.88 | 1.00 |
| Urban | 66 | 8 / 12 | 0.83 | 0.85 | 1.00 |

Every group's struggling students are all found. Boys are flagged more often
(28 % against 17 %) because more of them truly need support (20 % against
12 %). The model's input cannot stand in for sex: G1 predicts sex with ROC-AUC
0.555, close to chance. With 32–66 students per group, accuracy gaps of about
0.05 are within chance.

### 6.6 Cost of leaving sex out

Trained with sex allowed, RFECV again kept only G1, so the model and its test
scores are identical (balanced accuracy 0.879, Needs-support recall 1.000). In
this setting the fairness choice costs nothing.

### 6.7 Error analysis (test set)

17 of 98 students are in the wrong group, none of them two groups off. In 12 of
the 17 the network predicted the group of G1. The misclassified students moved
+1.0 points from G1 to G3 on average, against +0.43 overall: they sit near a
boundary and crossed it (G1 = 8 who finished on 10–11, G1 = 14 who slipped to
12–13). With G1 as the only input they look identical to students who did not
move, so no model built on this input can separate them.

## 7. Web application

`python -m streamlit run app/Home.py` opens an early-warning board with five
pages under a top navigation bar:

| Page | What it shows |
|---|---|
| Board | the three groups, test results against the naive rule, and a watch list of the 15 test students with the highest chance of needing support, beside what really happened |
| Check a student | enter G1; the predicted group in its colour, a sentence worded by confidence (≥ 70 % "Likely", ≥ 50 % "Probably", else "Unclear"), one segmented meter per chance, and what pushed the prediction |
| Model report | the full test and cross-validation tables, significance tests, confusion matrices, ROC curves |
| What drives predictions | permutation importance |
| Experiments | repeated CV, ablation, learning curve, calibration, fairness, errors |

The design follows its use: a dark board a head of year scans, a lamp colour
per group (red Needs support, blue-grey On track, green High achiever) and
meters of 20 cells of 5 %. The watch list reproduces the test results: the
network flags 22 students, 15 of whom did finish below 10, and it catches all 15
who did. Every page stops with the command to run if the model is not trained.

## 8. Discussion and limitations

**Why the final network has one input.** This is the result most likely to be
questioned, so the evidence is collected here. RFECV's score was flat from 1 to
7 encoded columns (0.764 to 0.772) and fell after that; the one-standard-error
rule took the simplest of the equal options. The ablation then tested the
alternative directly on the network: RFECV's best count scored 0.736 and all
inputs 0.718, against 0.753 for G1 alone. Earlier probes on the same folds gave
the same picture: G1 with failures 0.762, a difference inside the fold spread;
all 30 inputs 0.705. On 551 students the background details are mostly noise
to the network once G1 is known. Forcing them in to make the model look richer
would make it worse.

**What the network adds over the rule.** Not a better label: the rule is as
accurate. The network turns G1 into three calibrated probabilities, and these
rank students by risk better than the rule in every one of 15 folds. A teacher
who can see only some students first needs that order, not just a label.

**Limitations.**

- **Small data.** 551 students for model selection and 98 for testing; scores
  move by a few points with another sample, so the intervals matter.
- **The test set is easier than average**, which flatters every model. The
  cross-validation numbers are the realistic ones.
- **Selection effect.** The design choices were compared on the same folds, so
  choosing the best of them is itself slightly optimistic.
- **One subject, two schools, one cohort** (Portugal, 2008 study). Nothing here
  shows the model transfers to other schools.
- **Ordered groups treated as unordered classes.** An ordinal loss would use the
  order.
- **G1 carries earlier inequality.** A model built on it carries that forward.
  The tool should prompt a teacher to look, never decide about a student, and
  should be re-audited on every new cohort.

## 9. Conclusion and future work

After the first term, the first-term grade carries almost all the information
this dataset holds about the final group. The deep network, chosen by
leakage-free cross-validation, matches the naive "group of G1" rule on the
label, finds every student who needs support on the test set, and ranks
students by risk better than the rule in all 15 cross-validation folds. Leaving
sex out costs nothing, and every audited group's struggling students are
found.

Next steps:

1. Data from more schools and years, to test transfer and give the background
   details a fair chance to help.
2. An ordinal loss that uses the order of the three groups.
3. Re-test G1 with failures, the only addition that came close, on new data.
4. Monitor calibration and fairness on each new cohort.

## References

- Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
- Cortez, P. and Silva, A. (2008). Using data mining to predict secondary school student performance. *Proceedings of the 5th Future Business Technology Conference*, 5–12.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
- Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Guo, C. and Berkhahn, F. (2016). Entity embeddings of categorical variables. arXiv:1604.06737.
- Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Guyon, I., Weston, J., Barnhill, S. and Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46, 389–422.
- Ioffe, S. and Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. *ICML*.
- Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*.
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A. (2018). Hyperband: a novel bandit-based approach to hyperparameter optimization. *JMLR*, 18(185), 1–52.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K. and Dollár, P. (2017). Focal loss for dense object detection. *ICCV*.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15, 1929–1958.
