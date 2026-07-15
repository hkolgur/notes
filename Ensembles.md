# Ensembles (Bagging, Boosting, Stacking, Cascading) — Interview Notes

## 1. What is an Ensemble?

An **ensemble** combines multiple models ("base learners") so the group predicts
better than any single model — the ML version of "wisdom of crowds."

**Key requirement: diversity.** The more *different* (decorrelated) the base
models' errors are, the more the combination helps. Uncorrelated mistakes cancel
out when aggregated; identical models add nothing.

Four main families:
1. **Bagging** (Bootstrap Aggregation) — e.g. Random Forest
2. **Boosting** — e.g. AdaBoost, GBDT, XGBoost
3. **Stacking** (and Blending)
4. **Cascading**

---

## 2. Bias–Variance Decomposition (the "why" of ensembles)

```
Expected Error = Bias² + Variance + Irreducible error (noise)
```

> ⚠️ The third term is **irreducible error** (noise inherent in the data),
> not "generalization error."

- **High variance model:** changes a lot when the training data changes slightly
  (e.g. a fully grown decision tree) → overfits
- **High bias model:** too simple to capture the pattern (e.g. a depth-1 stump)
  → underfits

The two ensemble strategies attack different terms:

| | Bagging | Boosting |
|---|---|---|
| Attacks | **Variance** | **Bias** |
| Base learners should be | **Low bias, high variance** (deep/fully-grown trees) | **High bias, low variance** (shallow trees / stumps) |
| Training | **Parallel** (models independent) | **Sequential** (each model fixes the previous one's errors) |
| Aggregation | Majority vote / mean | Weighted additive sum |
| Examples | Random Forest, ExtraTrees | AdaBoost, GBDT, XGBoost, LightGBM, CatBoost |

---

## 3. Bagging (Bootstrap Aggregation)

### Procedure
1. From training data D (n points), draw k **bootstrap samples** D1, …, Dk —
   each of size n, **sampled with replacement**
2. Train one model Mi per sample (in parallel)
3. **Aggregate:** majority vote (classification) or mean/median (regression)

### Why it reduces variance
Any single point influences only some of the bootstrap samples, so
adding/removing a few points changes only a few base models — the aggregate
barely moves. Averaging k models with (partially) independent errors shrinks
variance roughly like averaging noisy measurements, **without increasing bias**
(the final bias ≈ base-model bias).

Result with deep-tree bases: `low bias + high variance` bases → aggregated model
is `low bias + low variance`.

### Bootstrap math (interview favorite)
Probability a given point is NOT picked in one draw = (1 − 1/n).
Not picked in n draws = (1 − 1/n)ⁿ → **1/e ≈ 36.8%** as n grows.

- Each bootstrap sample contains ≈ **63.2%** unique points ("in-bag")
- ≈ **36.8%** are left out → **Out-of-Bag (OOB) points**, usable as a free
  validation set for that model (see Random Forest notes)

---

## 4. Stacking

### Idea
Train **heterogeneous** base learners (e.g. logistic regression + SVM + NN + tree)
in parallel, then train a **meta-model** (often logistic regression) whose inputs
are the base models' predictions:

```
Level 0:  h1(x), h2(x), …, hm(x)        (diverse base models)
Level 1:  meta(h1(x), h2(x), …, hm(x))  → final prediction
```

Using **predicted probabilities** (not hard labels) as meta-features usually
works better — the meta-model sees confidence, not just votes.

### Avoiding leakage — Stacking vs Blending (get this right!)
The meta-model must be trained on predictions for data the base models did NOT
train on, or it just learns to trust overfit outputs.

- **Stacking (k-fold / out-of-fold):** split training data into k folds; for each
  fold, train base models on the other k−1 folds and predict the held-out fold.
  Concatenating these **out-of-fold (OOF) predictions** covers the whole training
  set → meta-model trains on all of it. More data-efficient, more robust.
- **Blending (holdout):** simpler variant — train base models on e.g. 80% of the
  data, predict on the remaining 20% holdout, and train the meta-model only on
  that holdout's predictions. Faster but the meta-model sees less data.

> ⚠️ Common notes error: these two definitions often get swapped.
> **k-fold OOF = stacking; single holdout = blending.**

### Practical notes
- Available in **sklearn** as `StackingClassifier` / `StackingRegressor`
  (since v0.22); `mlxtend` also provides `StackingClassifier`
  (older resources saying "not in sklearn" are outdated)
- Dominant in **Kaggle competitions** (squeezes the last % of accuracy)
- In production the cost is high: many models to train, serve, and maintain —
  latency and maintenance often outweigh the small accuracy gain

### Stacking vs Bagging/Boosting
- Bagging & boosting use **homogeneous** base learners (all trees);
  stacking uses **heterogeneous** learners
- Bagging/boosting aggregate by vote/sum; stacking **learns** the combination

---

## 5. Cascading 🔍 *(rarely a named interview topic — the concept survives in
system-design questions like "design a fraud-detection pipeline")*

Used when the **cost of a mistake is very high** (credit-card fraud, medical
screening). Chain models with confidence gates: M1 handles the easy ~99%
(e.g. predict "not fraud" only if P > 0.99); uncertain cases pass to M2 — trained
only on points M1 was unsure about — then M3, …, and finally a **human** decides.
Cheap models handle most traffic; expensive stages see only hard cases.

Vs stacking: stacking sends **every** query through all models and learns to
combine them; cascading gates queries **sequentially by confidence**, so most
never reach later models. (🔍 Trivia: Viola–Jones face detection is a cascade of
AdaBoost classifiers — CV-role material only.)

---

## 6. Choosing an Ensemble — quick guide

| Situation | Use |
|---|---|
| Single tree overfits (high variance) | **Bagging / Random Forest** |
| Model underfits, need accuracy (bias problem) | **Boosting (GBDT/XGBoost)** |
| Several strong-but-different models available; competition setting | **Stacking** |
| Mistakes are very costly; need high-confidence decisions | **Cascading** |
| Need parallel/fast training | Bagging (boosting stages are sequential) |

---

## 7. Frequently Asked Interview Questions

1. Write the bias–variance decomposition. Which term does bagging reduce and
   which does boosting reduce? What base learners does each therefore prefer?
2. Why does bagging reduce variance without (much) affecting bias?
3. Prove ≈63.2% of points appear in a bootstrap sample. What are OOB points and
   what are they used for?
4. Why must base models in an ensemble be diverse/decorrelated? What happens if
   they are identical?
5. Bagging vs boosting — parallel vs sequential, and why.
6. Explain stacking. Why must the meta-model be trained on out-of-fold
   predictions (what leakage occurs otherwise)?
7. Stacking vs blending — precise difference.
8. Why use predicted probabilities instead of hard labels as meta-features?
9. Why is stacking common in Kaggle but rare in production?
10. Describe cascading. When is it preferred, and how does it control cost/latency?
11. 🔍 (Rare/CV roles) How does Viola–Jones face detection use a cascade?
12. (System-design framing) Design an ensemble for credit-card fraud detection
    with a human-in-the-loop — this is how cascading actually gets asked.
