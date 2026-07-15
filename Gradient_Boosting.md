# Boosting: AdaBoost, GBDT, XGBoost, LightGBM, CatBoost — Interview Notes

## 1. Boosting — Core Idea

Build an **additive model in stages**, where each new weak learner corrects the
errors of the ensemble so far:

```
F0(x) = simple initial guess
Fm(x) = Fm-1(x) + ν · hm(x)        (m = 1 … M)
Final: FM(x) = F0(x) + ν·Σ hm(x)
```

- Base learners are **weak: high bias, low variance** (shallow trees; leaves
  typically 8–32 for GBDT, stumps for AdaBoost)
- Boosting **reduces bias** stage by stage (bagging reduces variance)
- Training is inherently **sequential** — stage m needs Fm-1's errors

---

## 2. GBDT by Hand — Regression Walkthrough
*(the StatQuest-style example: predicting weight)*

1. **F0 = mean of target.** With squared loss, the constant minimizing the loss
   is the mean (e.g. every person's predicted weight starts at 171.8).
2. **Compute residuals:** ri = yi − F0(xi) (e.g. 88 − 171.8 = −83.8, …).
3. **Fit tree h1 on {xi, ri}** — features predict the residuals, not y.
4. **Update with learning rate:** F1(x) = F0(x) + 0.1·h1(x)
   (prediction 171.8 moves a *small step* toward the truth, e.g. +0.1×3.5).
5. **Repeat:** new residuals yi − F1(xi) are smaller; fit h2 on them; continue
   for M trees.
6. **Predict:** F0 + ν·(h1 + h2 + … + hM).

Small steps (ν≈0.1) with many trees generalize far better than one big jump —
this is the whole point of shrinkage.

---

## 3. Pseudo-Residuals — Gradient Descent in Function Space

For **squared loss** L = (y − F(x))², the residual y − F(x) *is* (proportional to)
the **negative gradient** of the loss w.r.t. the prediction:

```
−∂L/∂F(x) = 2(y − F(x))  ∝  residual
```

Generalization (Friedman): at each stage fit the tree to the **pseudo-residuals**

```
rim = − [ ∂L(yi, F(xi)) / ∂F(xi) ]  evaluated at F = Fm-1
```

**Why this matters (the crux):** GBDT can minimize **any differentiable loss** —
squared/absolute/Huber loss for regression, logistic loss for classification,
ranking losses, custom losses. (Contrast: RF just optimizes impurity; it can't
target an arbitrary loss like hinge/logistic.)

### Friedman's Gradient Boosting Algorithm 🔍 *(Deep-dive — interview-level answer
is just "fit each tree to the negative gradients of the loss, scale by learning
rate." The per-leaf line search below is senior/research-round depth.)*
```
1. F0(x) = argmin_γ Σ L(yi, γ)            (for squared loss → γ = mean of y)
2. For m = 1 … M:
   a. rim = −∂L(yi, F(xi))/∂F(xi) |F=Fm-1      (pseudo-residuals)
   b. Fit a regression tree to {(xi, rim)} → terminal regions Rjm
   c. For each leaf j: γjm = argmin_γ Σ(xi∈Rjm) L(yi, Fm-1(xi) + γ)   (line search)
   d. Fm(x) = Fm-1(x) + ν · Σj γjm · 1(x ∈ Rjm)
3. Output FM(x)
```

---

## 4. Regularization

- **Shrinkage / learning rate ν (0 < ν ≤ 1, typically 0.1):** scales each tree's
  contribution. Smaller ν → less overfitting/variance, but needs larger M.
  **M and ν trade off** — tune jointly with cross-validation.
- **Subsample < 1 (Stochastic Gradient Boosting):** each tree fits a random
  fraction of rows → adds bagging-style variance reduction.
- **Tree constraints:** max_depth (default 3 in sklearn), min_samples_leaf.
- **More trees CAN overfit** (unlike RF!) — M is a real capacity knob; use early
  stopping on a validation set.

### Complexity
- Train: O(n log n · M) — **stages are sequential**, not parallelizable across
  trees (modern libraries do parallelize *within* a tree's split search)
- Test: O(depth · M) — shallow trees → very low latency
- Space: O(trees + leaf values γ)
- sklearn defaults: `learning_rate=0.1`, `max_depth=3`, `n_estimators=100`

---

## 5. AdaBoost (Adaptive Boosting)

The original boosting algorithm — reweights **points** instead of fitting residuals.

1. Initialize equal weights wi = 1/n (weights always sum to 1)
2. Train a weak learner (a **stump**: root + 2 leaves); compute weighted error ε
3. Compute the model's **"amount of say"**: α = ½·ln((1−ε)/ε)
4. **Increase weights of misclassified points** (×e^α), decrease weights of
   correct ones (×e^−α); renormalize to sum 1
5. Next stump focuses on the hard (high-weight) points; repeat
6. Final prediction = sign(Σ αm·hm(x)) — a **weighted** vote

Classic application: **face detection (Viola–Jones)** in computer vision,
combined with a cascade.

### AdaBoost vs Gradient Boosting
| | AdaBoost | Gradient Boosting |
|---|---|---|
| Corrects errors via | **Re-weighting** misclassified points | Fitting **pseudo-residuals** (negative gradients) |
| Base trees | Stumps (1 split) | Shallow trees (≈8–32 leaves) |
| Loss | Exponential loss (implicitly — AdaBoost is GB with exponential loss) | Any differentiable loss |
| Model weights | Unequal "say" α per stump | Uniform ν·γ (learned leaf values) |

### AdaBoost vs Random Forest
1. AdaBoost = boosting; RF = bagging
2. AdaBoost sequential (stage m depends on m−1); RF trees independent/parallel
3. AdaBoost uses stumps; RF grows full trees
4. AdaBoost models have unequal say (α); RF trees vote equally

---

## 6. XGBoost

**XGBoost ≈ GBDT + row sampling + column sampling + heavy regularization +
systems engineering** (it borrows RF's sampling ideas):

- **Regularized objective:** loss + Ω(tree) where Ω = γ·(#leaves) + ½λ·Σ(leaf weights)²
- 🔍 *Deep-dive:* uses **second-order Taylor expansion** (gradients *and*
  hessians) for split gain — senior-ML-engineer follow-up; the expected answer is
  the regularization + missing-value + sampling line above
- **Sparsity-aware:** learns a **default direction** per split for missing values
- Parallelized/cache-optimized split finding, histogram approximation
- Released 2014; huge community; `objective='reg:squarederror'`,
  `'binary:logistic'`, etc.; supports custom objectives

> Relation to linear models: gradient boosting is gradient descent in function
> space — same descent idea as training linear regression, but each "step" is a
> whole tree instead of a parameter update.

---

## 7. XGBoost vs LightGBM vs CatBoost

| | **XGBoost** (2014) | **LightGBM** (Microsoft, 2016) | **CatBoost** (Yandex, 2017) |
|---|---|---|---|
| Tree growth | **Level/depth-wise** — grows whole levels; different conditions per node | **Leaf-wise** — repeatedly split the single leaf with max gain → asymmetric trees | **Symmetric (oblivious)** — same split condition for all nodes at a level |
| Categorical features | Encode manually (one-hot/target); newer versions have experimental native support | Native — bin/bucket-based grouping of categories | Native — **Ordered Target Encoding** (leakage-safe); `one_hot_max_size` controls small categories |
| Sampling | Row/column subsample (without replacement) | **GOSS** — Gradient-based One-Side Sampling | Minimal-variance / uniform sampling; 🔍 Ordered Boosting (internals rarely asked) |
| Speed tricks | Histogram, parallel split search | Histogram **binning**, **EFB** (Exclusive Feature Bundling) | Symmetric trees → very fast inference; strong GPU support |
| When to pick | Max control/tuning, community support, robust all-rounder | **Speed** on large data; good results with little tuning | **Many categorical features**; large data; GPU |

### LightGBM speed tricks (know these!)
1. **Histogram binning:** bucket continuous values (e.g. salary → 50–80k, 80–90k)
   and search splits over bins, not raw values
2. **EFB (Exclusive Feature Bundling):** mutually exclusive sparse features
   (e.g. one-hot columns where only one is nonzero) get bundled into a single
   feature → fewer columns
3. **GOSS:** keep the **large-gradient points** (poorly predicted → most to
   learn from) and randomly sample only a fraction of the small-gradient rest
   (already well fit). 🔍 *Deep-dive:* the sampled small-gradient points are
   re-weighted by (1−a)/b to keep gradients unbiased — implementation trivia,
   rarely asked
4. **Leaf-wise growth:** spend splits only where loss reduction is largest
   (deeper, asymmetric trees; cap `num_leaves`/depth to avoid overfitting)

---

## 8. Frequently Asked Interview Questions

1. Boosting vs bagging: which term of the error does each reduce, and what base
   learners does each use? Why sequential vs parallel?
2. Walk through GBDT for regression by hand: initial prediction, residuals,
   learning-rate update, final prediction.
3. Show that for squared loss the residual is the negative gradient. What is a
   pseudo-residual and why does it let GB minimize any differentiable loss?
4. 🔍 (Senior rounds) Write/explain Friedman's gradient boosting algorithm (init,
   pseudo-residuals, fit tree, leaf line-search, shrinkage update). Standard
   rounds only expect the one-line version from Q3.
5. What does the learning rate (shrinkage ν) do? Describe the M–ν trade-off and
   how you'd tune both.
6. Can gradient boosting overfit as you add trees? How do you detect and prevent
   it? (Contrast with RF.)
7. How does AdaBoost work: weight updates, amount of say α = ½ln((1−ε)/ε),
   final weighted vote. How does it relate to GB? (Exponential loss.)
8. AdaBoost vs GBDT vs Random Forest — a three-way comparison.
9. What does XGBoost add on top of vanilla GBDT? (Regularization Ω, second-order
   gradients, sparsity-aware missing handling, sampling, systems optimizations.)
10. How does XGBoost handle missing values at split time?
11. Level-wise vs leaf-wise vs symmetric tree growth — which library does which,
    and the overfitting/speed implications of each.
12. Explain GOSS and EFB. Why do they make LightGBM fast without hurting accuracy?
13. What is CatBoost's ordered target encoding, and what leakage problem does it
    solve compared to naive target/response encoding?
14. You have a dataset with 200 high-cardinality categorical columns — which
    booster do you reach for and why?
15. Why is boosting hard to parallelize across trees, and what do modern
    libraries parallelize instead?
16. Train/test complexity of GBDT; why is it good for low-latency serving despite
    having M trees?
