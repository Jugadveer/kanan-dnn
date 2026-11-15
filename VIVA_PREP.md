# Viva preparation: Early warning after the first term

Everything a strict examiner can ask about this project, with the answer and
where it is in the code. The interview scores five areas, 1 to 5 each:

| Area | What it tests | Sections |
|---|---|---|
| P1 Core Architecture | embeddings, the functional network, focal loss, BatchNorm, L2, ensemble, calibration | 6, 7, 8 |
| P2 Problem Solving | framing, the naive baseline, leakage, model selection, honesty about results | 2, 3, 4, 5, 9, 10, 11 |
| P3 Tech Stack | Keras 3, KerasTuner, scikit-learn, SciPy, Streamlit, YAML, pytest | 13 |
| P4 System Design | CLI, config, artifacts, app, tests | 12 |
| P5 Tech Communication | the pitch, the questions | 1, 15, 16 |

Study each section with its code file open.

---

## 1. The pitch

**30 seconds.** "Right after the first term, my network estimates whether a
student's final Portuguese grade will be Needs support, On track or High
achiever. I compared it honestly with the obvious rule, 'final group = group
of the first-term grade', which is already right for 75 % of students. On 98
unseen students the network found all 15 who needed support, and it ranks
students by risk better than the rule in every cross-validation fold (ROC-AUC
0.883 against 0.801). On the hard label it ties the rule (0.753 against 0.752
balanced accuracy, McNemar p = 0.45). So after the first term, the first-term
grade carries almost all the signal; what the network adds is calibrated
probabilities that put students in order of risk."

**2 minutes.** Add the method: G1 in, G2 and sex out; stratified 70/15/15
split; RFECV with the one-standard-error rule (kept only G1); Keras functional
network with embedding branch, BatchNorm, dropout, L2, α-balanced focal loss;
Hyperband proposed 90 networks, 5-fold CV on 551 students chose among the top
three; a 5-seed ensemble was tested and rejected by CV; temperature scaling on
validation; test set used once with bootstrap intervals and McNemar's test;
experiments for ablation, learning curve, calibration, fairness and errors.

## 2. Problem framing

- **When:** right after the first term. Known: the first-term grade G1 and the
  student's background. Not known: G2 (second term) and G3 (final, the target).
- **Why not G2?** It arrives at the end of the second term; a warning then
  leaves only one term to act. With G1 the warning comes with two terms left.
- **Why not sex?** An earlier version of this model (without G1) relied on sex
  more than on anything else. Predicting a student's risk partly from their sex
  is unfair to the individual, so it is never an input (`config.yaml`,
  `inputs.left_out`). The experiments measure the cost.
- **Target:** three ordered groups of G3: Needs support 0-9 (100 students,
  15 %), On track 10-13 (355, 55 %), High achiever 14-20 (194, 30 %).
  `category_of` in [loading.py](earlywarning/loading.py) uses
  `np.searchsorted([9, 13, 20], grade)`.
- **The bar to clear:** the naive rule "final group = group of G1" is right
  for 488 of 649 students (75.2 %). A model is only useful if it adds to that.

## 3. Data and EDA

- UCI Student Performance (Cortez and Silva, 2008), Portuguese course:
  649 students, 33 columns, 0 missing, 0 duplicates, separator `;`.
- Inputs offered: 30 (30 attributes + G1 − sex). Groups of the brief in
  `columns.yaml`: Demographic, Academic, Social, Behavioural.
- **G1 against G3** (`python -m earlywarning explore`):

  | Group of G1 \ final group | Needs support | On track | High achiever |
  |---|---|---|---|
  | Needs support (G1 0-9) | 88 | 69 | 0 |
  | On track (G1 10-13) | 12 | 267 | 61 |
  | High achiever (G1 14-20) | 0 | 19 | 133 |

  Students improve on average (+0.51 points from G1 to G3). The naive rule's
  main mistake: 69 students with G1 below 10 who still passed.
- Spearman correlation with G3: G1 +0.88, failures −0.45, Medu +0.28,
  studytime +0.27, Fedu +0.23, Dalc −0.21. Spearman uses ranks, so it suits
  ordered codes and any monotone relationship.

## 4. Preprocessing and leakage

- **Flat encoding** (KNN, tree, Logistic Regression in RFECV): standardise
  numbers, one-hot encode text (`flat_encoder` in [encoding.py](earlywarning/encoding.py)).
- **Network encoding** (`EmbeddingEncoder`): integer codes per text column,
  shifted so 0 means "never seen in training"; numbers standardised.
- **Leakage** = information from the students a model is scored on reaching its
  training. Prevented by: splitting the test set off first; fitting every
  encoder, every RFECV and every alpha only on the rows a model trains on;
  refitting RFECV inside every CV fold ([validation.py](earlywarning/validation.py)).
  Tests check that encoder statistics come from training rows only.

## 5. Feature selection: RFECV + one-standard-error rule

Code: [selection.py](earlywarning/selection.py).

- **RFE:** fit Logistic Regression, drop the encoded column with the smallest
  coefficients, repeat. **CV:** score every column count by 5-fold balanced
  accuracy.
- **One-standard-error rule** (Breiman et al., 1984): take the fewest columns
  whose score is within one standard error (std / √5) of the best. Smaller
  differences are noise; the simpler model is preferred.
- **Why:** with RFECV's best count the network scored 0.717 in CV; with the
  rule it scores 0.753. Extra background columns were mostly noise for the
  network on 551 students. (Probes: all 30 inputs 0.705; G1 + failures 0.762.)
- **Result on the training split:** RFECV's best score came with 2 encoded
  columns; the rule kept 1 of 55: **G1**. In the 5 CV folds it kept G1 alone
  four times, and G1 + age, Fjob, failures, higher once.
- So after the first term, the first-term grade carries nearly all the usable
  signal. Be ready to defend this: it is what the data says, measured without
  leakage, and the ablation shows adding the columns back does not help.

## 6. The network (P1)

Code: `build_network` in [model.py](earlywarning/model.py), Keras functional API.

```
text columns  -> one Embedding each --\
numbers (G1, ...) -> standardised -----+-> Concatenate -> [Dense (no bias) -> BatchNorm -> ReLU -> Dropout] x N -> Dense 3 softmax
```

The final network (chosen by CV): input G1 (standardised, 1 number) → Dense 64
→ Dense 128 → Dense 16, each with BatchNorm, ReLU, dropout 0.4, L2 0.01 →
Dense 3 softmax. Learning rate 0.001.

| Layer | Calculation | Trainable weights |
|---|---|---|
| Dense 64 (no bias) | 1 × 64 | 64 |
| BatchNorm 64 | 2 × 64 | 128 |
| Dense 128 (no bias) | 64 × 128 | 8,192 |
| BatchNorm 128 | 2 × 128 | 256 |
| Dense 16 (no bias) | 128 × 16 | 2,048 |
| BatchNorm 16 | 2 × 16 | 32 |
| Output Dense 3 | 16 × 3 + 3 | 51 |
| **Total** | | **10,771** |

Is that too big for one input? It is what CV chose among Hyperband's best;
dropout 0.4, L2 0.01 and early stopping (191 epochs run) keep it smooth. With
one input the network is a flexible curve from G1 to three probabilities,
which is exactly what makes its probabilities rank better than the rule.

- **Entity embedding** (Guo and Berkhahn, 2016): a learnt lookup table, one
  small vector per value; rows = values + 1 ("unseen"). With only G1 kept there
  are no text inputs, so the embedding branch is empty; it is used when
  selection keeps text columns (one CV fold kept Fjob).
- **Dense layers without bias:** BatchNorm subtracts the batch mean straight
  after, which cancels a bias, then adds its own learnt shift β.
- **Batch normalisation** (Ioffe and Szegedy, 2015): x̂ = (x − batch mean) /
  √(batch variance + ε), output γ·x̂ + β. Uses running averages at prediction.
- **ReLU** max(0, x); **dropout** zeroes units at random during training only
  (Srivastava et al., 2014); **L2** adds λ·Σw² to the loss.
- **Softmax** gives three probabilities that sum to 1.
- **Parameter count:** Dense(in, out) has in × out weights (+ out biases if
  used); BatchNorm has 2 × units trainable (γ, β) and 2 × units running statistics.

## 7. Training, focal loss, ensemble, calibration

- **α-balanced focal loss** (Lin et al., 2017): FL = −α_c (1 − p)^γ log p,
  γ = 2. p = 0.9 → factor 0.01 (easy student nearly ignored); p = 0.3 →
  0.49. α_c = n / (3 · n_c) from the training labels: 2.157 Needs support,
  0.611 On track, 1.110 High achiever.
- **Adam** (Kingma and Ba, 2015), batch 32; **early stopping** on validation
  loss, patience 25, best weights restored.
- **Seed ensemble:** the same network trained with 5 seeds, probabilities
  averaged. Kept only if CV says it helps: it scored 0.750 against 0.753 for
  one network, so the final model is a single network.
- **Temperature scaling** (Guo et al., 2017): softmax(log p / T), T fitted on
  the validation students by minimising log loss. T < 1 sharpens, T > 1 softens;
  the predicted group never changes. Here T = 0.52: focal loss pulls
  probabilities towards the middle (it stops rewarding confidence once p is
  high), so the raw network is under-confident and scaling sharpens it.
- **Reproducibility:** fixed seeds and deterministic TensorFlow ops. Two full
  training runs produced identical results.json (test scores, comparisons,
  selection, network, model selection, Hyperband trials).

## 8. Model selection

- **Hyperband** (Li et al., 2018): many candidates on a small epoch budget,
  keep the best third, triple the epochs: budgets 2, 4, 10, 30; 90 trials,
  scored on the 98 validation students.
- **Why not trust Hyperband's winner?** 98 students make a noisy score. In an
  earlier version the winner scored 0.652 on validation but 0.486 on test.
  Now Hyperband only proposes; 5-fold CV on all 551 pool students decides:

  | Candidate (units, dropout, L2, learning rate) | Hyperband (98) | CV (551) |
  |---|---|---|
  | **64-128-16, 0.4, 0.01, 0.001** | 0.813 | **0.753 ± 0.033** |
  | 128-16-32, 0.3, 0.0001, 0.0003 | 0.813 | 0.752 ± 0.034 |
  | 64-32-64, 0.5, 0.0001, 0.001 | 0.813 | 0.743 ± 0.040 |

  Hyperband gave all three the same balanced accuracy (0.813), so on 98
  students it could not separate them; CV separated them a little. Note the gap: 0.813 on validation, 0.753 in CV.
- **Baselines on the same folds:** KNN grid (k 3-35, uniform/distance), tree
  grid (depth 2-8, leaf 1-20). Chosen: KNN k = 9 with distance weights (CV
  0.728); tree depth 2, leaf 1 (CV 0.747). A depth-2 tree on G1 is two cut-offs,
  almost the naive rule with learnt boundaries.

## 9. Evaluation and results

- **Three questions:** which group (balanced accuracy = mean recall over the
  three groups; macro F1; one-vs-rest ROC-AUC); needs support or not (recall,
  precision, ROC-AUC, accuracy); at most one group off.
- **On the test set within-one is 1.000 for every trained model, even "always On track"**: with three
  ordered groups only a Needs support ↔ High achiever confusion counts, and no
  model makes one. Say so if asked; it is not an achievement.
- **95 % bootstrap intervals** (Efron and Tibshirani, 1993): resample the 98
  test students with replacement 2000 times, score each, take the 2.5th and
  97.5th percentiles.
- **McNemar's exact test** (McNemar, 1947): count test students only model A
  gets right (b) and only B gets right (c); if equally good, b ~ Binomial(b + c, ½).
- **Paired CV folds:** both models on the same folds; mean difference, folds
  won, paired t-test (optimistic, Dietterich, 1998).

**Test set (98 students, used once):**

| Model | Balanced acc. | Macro F1 | ROC-AUC | NS recall | NS precision |
|---|---|---|---|---|---|
| Neural network | 0.879 (0.82-0.93) | 0.825 | **0.946** | **1.000** | 0.682 |
| Group of G1 (naive) | **0.887** (0.82-0.94) | **0.850** | 0.906 | **1.000** | 0.682 |
| Decision tree | 0.817 | 0.824 | 0.903 | 0.733 | 0.733 |
| KNN | 0.817 | 0.824 | 0.929 | 0.733 | 0.733 |
| Always On track | 0.333 | 0.237 | 0.500 | 0 | 0 |
| Random | 0.274 | 0.273 | 0.459 | 0.067 | 0.071 |

(NS = Needs support. Network on the yes/no question: ROC-AUC 0.974, accuracy 0.929.)

**Cross-validation (5 folds, 551 students):** network 0.753 ± 0.033 balanced
accuracy, naive 0.752 ± 0.031, tree 0.747, KNN 0.728; ROC-AUC network 0.883,
tree 0.856, KNN 0.845, naive 0.801.

**Significance:**

| Network against | Test: only network right / only rival | McNemar p | CV balanced acc. diff (folds) | CV ROC-AUC diff (folds) |
|---|---|---|---|---|
| Naive rule | 2 / 5 | 0.453 | +0.001 (4 of 5) | **+0.082 (5 of 5)**, t-test p 0.0005 |
| Decision tree | 6 / 8 | 0.791 | +0.006 (2 of 5) | **+0.027 (5 of 5)**, t-test p 0.014 |

**How to say it:** "On the label, the network and the first-term rule are
statistically indistinguishable; the rule even has the higher macro F1 in 4 of 5
folds. On ranking, the network is better in every fold. The test set is easier
than average (the rule scores 0.887 there but 0.752 in CV), which is why I lead
with cross-validation."

## 10. Experiments

Code: [experiments.py](earlywarning/experiments.py); run `python -m earlywarning experiments` (72 minutes measured).

**Repeated CV (5 folds × 3 repeats = 15 folds, RFECV inside each):**

| | Network | Naive rule | Decision tree |
|---|---|---|---|
| Balanced accuracy | **0.756** ± 0.048 | 0.752 ± 0.035 | 0.746 ± 0.038 |
| Macro F1 | 0.702 | 0.721 | **0.756** |
| ROC-AUC | **0.886** | 0.801 | 0.857 |
| Needs-support ROC-AUC | **0.940** | 0.863 | 0.923 |

Network vs naive over 15 folds: ROC-AUC +0.085 (ahead in 15 of 15); balanced
accuracy +0.003 (10 ahead, 5 behind, p = 0.64); macro F1 −0.019 (12 behind,
p = 0.009). Network vs tree: Needs-support recall +0.133 (13 ahead, 2 tied),
precision −0.196 (13 behind). **Say it this way:** better ranking everywhere; a
tie on balanced accuracy; worse macro F1 because the class weights buy recall on
rare groups with precision.

**Ablation (5-fold CV, one change at a time; reference 0.753):**

| Change | Balanced accuracy | Difference | Takeaway |
|---|---|---|---|
| Without G1 | 0.556 | **−0.197** | G1 is the model. |
| No feature selection | 0.718 | −0.035 | extra inputs are noise |
| No class weights | 0.728 | −0.025 | α matters for balanced accuracy |
| RFECV best count | 0.736 | −0.017 | 1-SE rule generalises better |
| Ensemble, γ = 0, no L2 | 0.750 each | −0.003 | no effect |
| One-hot | 0.754 | +0.001 | no text inputs in 4 of 5 folds |
| No BatchNorm | 0.758 | +0.005 | within noise |
| Sex added back | 0.753 | 0.000 | RFECV never picks sex |

Fold spread is about ± 0.03, so differences below about 0.01 mean "no effect".

**Learning curve:** network 0.678 (88 students) → 0.751 (176) → 0.765 (264) →
0.753 (440); naive rule flat at 0.752; tree 0.737 → 0.747. The network needs
about 176 students to match the rule and then plateaus: more of the same data
would not help; new information would.

**Calibration (test):** expected calibration error 0.220 → 0.140, Brier 0.372 →
0.325, log loss 0.583 → 0.498 after temperature scaling (T = 0.52, fitted on
validation). ECE = weighted average gap between confidence and accuracy over
confidence bins (5 bins here, about 20 students each). Brier (three groups) =
mean squared distance between probabilities and the one-hot truth.

**Error analysis (test):** 17 mistakes, none two groups off; 12 follow the
group of G1. Mistaken students moved +1.0 points from G1 to G3 on average
(+0.43 overall): they are students near a boundary who crossed it (G1 = 8 who
reached 10-11; G1 = 14 who slipped to 12-13). With G1 as the only input they
look identical to students who did not move.

## 11. Fairness and ethics

- **Sex is not an input** (it was the top input in the earlier start-of-year
  version). With sex allowed in this version, RFECV still keeps only G1:
  identical model and scores, so **the fairness choice costs nothing here**.
- **Audit (test set):** Needs-support recall is 1.00 for girls, boys, both
  schools, rural and urban. Boys are flagged more (28 % against 17 %) because
  more truly need support (20 % against 12 %). Can the inputs guess sex? ROC-AUC
  0.555, close to chance: G1 is not a proxy for sex.
- Groups have 32-66 test students, so accuracy gaps of about 0.05 (for example
  0.80 boys against 0.85 girls) are within chance; say so rather than over-claim.
- **Honest risk:** G1 itself reflects earlier advantages and disadvantages, so a
  model built on it carries them forward. The tool should prompt a teacher to
  look, never decide about a student; and it should be re-audited on new cohorts.

## 12. System design (P4)

```
config.yaml, columns.yaml, .streamlit/config.toml  -> settings.py
python -m earlywarning explore       EDA charts
python -m earlywarning train         Hyperband -> CV choice -> final fit -> calibrate -> test once
                                     -> artifacts/ (network_*.keras, encoder.joblib, serving.json, form.json)
                                     -> reports/results.json, importance.json
python -m earlywarning experiments   -> reports/experiments.json + charts
python -m pytest                     tests/
python -m streamlit run app/Home.py  Board, Check a student, Model report, What drives predictions, Experiments
```

- **App design:** drawn as a school's early-warning board, the screen a head of
  year scans: dark board, a lamp per group (red Needs support, blue-grey On
  track, green High achiever), segmented meters for each chance (20 cells of
  5 %), and on the Board page a watch list of the 98 test students ranked by
  their chance of needing support, with what really happened. The watch list
  reproduces the test recall (15 of 15 caught) and precision (15 of 22 flags).
  Colours and the Barlow typeface (drawn from road signs) are in
  `.streamlit/config.toml`, the rest in `.streamlit/board.css`.
- **One prediction:** an input changes → Streamlit reruns the page → inputs dict →
  `RiskPredictor.probabilities`: encoder → each network → average → temperature
  → three probabilities → banner (wording by confidence: ≥ 70 % "Likely",
  ≥ 50 % "Probably", else "Unclear: between ...") → "what pushed this
  prediction": each answer reset to the typical one, change in probability.
- **NetworkClassifier** makes the network look like a scikit-learn model
  (`fit`, `predict_proba`), so one cross-validation function serves every model.
- **Config in YAML**; **CLI** with subcommands, TensorFlow imported only when needed.
- **Missing model:** every page stops with the command to run.

## 13. Tech stack (P3)

| Tool | Version | Role |
|---|---|---|
| Python | 3.11 | language |
| TensorFlow / Keras | 2.21 / 3.15 | functional network, embeddings, focal loss |
| KerasTuner (+ TensorBoard) | 1.4.8 (2.21) | Hyperband; it imports TensorBoard on every trial |
| scikit-learn | 1.9 | RFECV/RFE, encoders, KNN, tree, dummies, metrics, CV |
| SciPy | 1.17 | binomial test (McNemar), paired t-test, temperature fitting |
| pandas / NumPy | 3.1 / 2.4 | tables, arrays |
| matplotlib | 3.11 | saved charts in the app's dark theme |
| Streamlit / Altair | 1.67 / 6.3 | multi-page app, interactive charts |
| PyYAML, joblib | 6.0, 1.6 | config files, saving the encoder |
| pytest | 9.2 | 24 tests, about 20 seconds |

## 14. Code reading order

1. `config.yaml`, `columns.yaml`, [settings.py](earlywarning/settings.py)
2. [loading.py](earlywarning/loading.py), [explore.py](earlywarning/explore.py)
3. [encoding.py](earlywarning/encoding.py), [selection.py](earlywarning/selection.py), [validation.py](earlywarning/validation.py)
4. [model.py](earlywarning/model.py) (every line), [calibration.py](earlywarning/calibration.py), [tuning.py](earlywarning/tuning.py)
5. [baselines.py](earlywarning/baselines.py), [evaluate.py](earlywarning/evaluate.py), [explain.py](earlywarning/explain.py)
6. [train.py](earlywarning/train.py), [experiments.py](earlywarning/experiments.py)
7. [predictor.py](earlywarning/predictor.py), [form.py](earlywarning/form.py), `app/`, `tests/`

## 15. Question bank

**Framing**
1. *Why after the first term?* It is the earliest point with a strong signal (G1) and still leaves two terms to help.
2. *Why not use G2?* It arrives too late to act on.
3. *Why remove sex?* Fairness: a student should not be judged partly by sex. The cost is measured in the experiments.
4. *Isn't a model that only uses G1 trivial?* The selection rule chose G1 alone because nothing else added accuracy in leak-free CV; I report that as a finding. The network still adds better-ranked, calibrated probabilities.
5. *What is the naive baseline and why include it?* "Final group = group of G1", right for 75 % of students. Without it, any model looks good.

**Data and selection**
6. *Class balance?* 15 / 55 / 30 %; handled with α in the loss and balanced accuracy.
7. *What is RFECV?* Drop the weakest column repeatedly; score every count by CV.
8. *What is the one-standard-error rule?* Fewest columns within one SE of the best score; differences below one SE are noise.
9. *Where could leakage happen and how did you stop it?* Encoders, RFECV and α could see held-out students; all are refitted inside every fold, and tests check it.

**Architecture and training**
10. *Draw the network and count the weights.* 1 → 64 → 128 → 16 → 3 with BatchNorm, ReLU, dropout 0.4; 10,771 trainable weights.
11. *What are embeddings for if only G1 is used?* They encode text columns when selection keeps them (one CV fold kept Fjob); here the branch is empty.
12. *Why no bias before BatchNorm?* BN removes the mean and adds its own shift β.
13. *Focal loss formula?* −α_c (1 − p)^γ log p, γ = 2.
14. *Why did the ensemble lose?* 0.750 against 0.753, a difference far inside the fold spread (± 0.033), so neither is really better and the simpler single network is kept. A likely reason: with one input, the five networks learn nearly the same curve, so averaging has little to smooth.
15. *Why temperature below 1?* Focal loss leaves the network under-confident; T = 0.52 sharpens its probabilities. The predicted group never changes.

**Model selection and evaluation**
16. *Why not trust Hyperband's best?* All three top candidates scored 0.813 on 98 validation students; CV on 551 separated them (0.753, 0.752, 0.743).
17. *Balanced accuracy?* Mean recall over the three groups.
18. *Why is within-one 1.000 for every trained model on the test set?* Only Needs support ↔ High achiever mistakes count, and none happens there. Over the 15 repeated-CV folds the network made one such mistake (0.999), the naive rule none.
19. *What does McNemar test?* Whether two models' errors on the same students differ; uses only students exactly one model gets right.
20. *Is the network better than the naive rule?* Not on labels: balanced accuracy ties (+0.003 over 15 folds, McNemar p 0.45 on test) and macro F1 is lower (behind in 12 of 15 folds). Yes on ranking: ROC-AUC +0.085, ahead in 15 of 15 folds.
21. *Why lead with CV, not the test set?* The test set is easier than average (naive 0.887 there, 0.752 in CV); 551 students in CV are more reliable than 98.
22. *Why is the paired t-test optimistic?* Folds share training students, so their scores are not independent (Dietterich, 1998).
23. *What is a bootstrap interval?* Resample the test students with replacement 2000 times; the middle 95 % of the scores.

**System, ethics, next steps**
24. *What does the app load?* network_0.keras, encoder.joblib, serving.json (temperature), form.json.
25. *How is "what pushed this prediction" computed?* Each answer reset to the typical one; the change in the predicted group's probability.
26. *How do you know runs are reproducible?* Two full training runs gave identical results.
27. *What would you do next?* More students or more schools; ordinal loss; use G1 + failures if more data shows it helps; monitor fairness on new cohorts.

## 16. Numbers to memorise

| What | Value |
|---|---|
| Students / columns / inputs offered | 649 / 33 / 30 (G1 in; G2, G3, sex out) |
| Groups | 100 / 355 / 194 (15 / 55 / 30 %) |
| Naive rule right | 488 of 649 (75.2 %) |
| Split | 453 / 98 / 98; pool 551 |
| Kept by RFECV + 1-SE | G1 only (1 of 55 encoded columns; best count was 2) |
| Network | 1 → 64 → 128 → 16 → 3, dropout 0.4, L2 0.01, lr 0.001, 10,771 weights |
| α (focal) / γ / T | 2.157, 0.611, 1.110 / 2 / 0.52 |
| Hyperband | 90 trials, top 3 all 0.813 on validation; CV 0.753 / 0.752 / 0.743 |
| CV balanced accuracy (5 folds) | network 0.753, naive 0.752, tree 0.747, KNN 0.728 |
| Repeated CV ROC-AUC (15 folds) | network 0.886, naive 0.801 (network ahead in 15 of 15) |
| Test, network | balanced acc 0.879 (0.82-0.93), macro F1 0.825, ROC-AUC 0.946, NS recall 1.000 |
| Test, naive | balanced acc 0.887 (0.82-0.94), macro F1 0.850, ROC-AUC 0.906 |
| McNemar network vs naive | 2 vs 5, p = 0.453 |
| Calibration (test) | ECE 0.220 → 0.140 |
| Ablation | without G1 −0.197; no selection −0.035; no α −0.025 |
| Fairness | NS recall 1.00 in every group; sex-from-inputs ROC-AUC 0.555 |
| Run times | train 34-51 min, experiments 72 min (CPU shared), tests about 20 s |

## 17. Before the viva

- [ ] `python -m earlywarning train` (34 to 51 minutes measured, CPU shared with other jobs) and `python -m earlywarning experiments` (72 minutes measured) at home, never in the room.
- [ ] `python -m pytest` passes.
- [ ] `python -m streamlit run app/Home.py`; demo: G1 = 8 (Needs support), then
      G1 = 10 and 11 (watch the wording move from "Likely" to "Unclear"), then the
      Model report table and McNemar row, then the ablation and fairness tables.
- [ ] Say the honest conclusion in one sentence: "After the first term the
      first-term grade carries almost all the signal; my network matches the
      first-term rule on the label and beats it at ranking students by risk."
