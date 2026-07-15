# Random Forest — Interview Notes

## 1. Definition

**Random Forest = Bagging with Decision Trees + Column (feature) Sampling**

```
RF = Base learners (deep DTs) + Row sampling (bootstrap) + Column sampling + Aggregation
```

Two sources of randomness decorrelate the trees:
1. **Row sampling (bagging):** each tree trains on a bootstrap sample
   (with replacement) — ≈63.2% unique points in-bag, ≈36.8% out-of-bag
2. **Column/feature sampling:** each **split** considers only a random subset of
   features — typically **√d** for classification, **d/3** for regression
   (feature subsets are drawn **without** replacement; "with replacement"
   applies only to the row bootstrap)

Aggregation: **majority vote** for classification; **mean** (sometimes median)
for regression.

---

## 2. Why does RF work? (Decorrelation — key interview story)

Example: predicting employee attrition with features {salary, age, commute}.
If salary is the strongest feature, **every** bagged tree would split on salary
first → trees are highly correlated → averaging correlated models barely reduces
variance.

With **column sampling**, some trees never see salary and are forced to learn
patterns from age/commute. These "different viewpoints" make tree errors less
correlated → averaging now genuinely cancels errors → the forest captures the
data's patterns "from various ends."

> Averaging k models reduces variance by ~1/k only if errors are independent;
> column sampling is what pushes tree errors toward independence.

---

## 3. Bias–Variance in RF

- Base trees are grown **deep / fully (even overfitting)** → each is
  **low bias, high variance**. That's fine — aggregation will fix the variance.
- **Final bias ≈ base-tree bias** (aggregation doesn't add bias)
- **Variance decreases as k (number of trees) increases** — and as row/column
  sampling rates decrease (more randomness → less correlation → lower variance,
  at some cost in individual-tree strength)
- **More trees never overfit** — test error plateaus as k grows; k is limited by
  compute, not overfitting. Tune k with CV or the OOB score.

---

## 4. Out-of-Bag (OOB) Evaluation

For each tree Mi, the ≈36.8% points not in its bootstrap sample (OOB points)
serve as a **free validation set**:

- Predict each training point using only the trees for which it was OOB
- Aggregate → **OOB score** ≈ cross-validation accuracy, with **no extra data
  split and no extra training**
- In sklearn: `oob_score=True`

---

## 5. Time & Space Complexity

| Phase | Cost | Notes |
|---|---|---|
| Train | O(n log n · d′ · k) | k trees; **trivially parallelizable** (`n_jobs=-1`) — trees are independent |
| Test | O(depth · k) | fast, but k× slower than one tree |
| Space (run) | O(nodes per tree · k) | store k trees |

RF's parallel training is a major practical advantage over boosting (whose
stages are sequential).

---

## 6. sklearn Hyperparameters

- `n_estimators` — number of trees k (more is better until plateau)
- `max_features` — features per split (`sqrt` default for classification)
- `max_depth` — usually left `None` (grow fully; bagging handles variance)
- `min_samples_split`, `min_samples_leaf` — per-tree regularization if needed
- `oob_score=True` — free validation estimate
- `class_weight='balanced'` — for imbalanced data
- `n_jobs=-1` — use all cores

---

## 7. Extremely Randomized Trees (ExtraTrees) 🔍 *(occasional one-liner, not core)*

Variation of RF with **even more randomness**:
- Instead of searching the **best threshold** for each candidate feature,
  pick thresholds **at random** and keep the best random split
- (sklearn's `ExtraTreesClassifier` also uses the whole dataset per tree by
  default — no bootstrap)

Effect: **faster training** (no threshold search), **lower variance**, slightly
**higher bias**. Useful when RF still overfits or training time matters.

---

## 8. Practical Considerations

- **Feature importance:** average the impurity-decrease importances over all k
  trees — more stable than a single tree's. Same caveats (cardinality bias,
  correlated features) → consider permutation importance.
- **Cases:** everything from the Decision Tree notes carries over (no scaling
  needed, native multi-class, no similarity-matrix input, axis-parallel
  boundaries, response-encode high-cardinality categoricals) — *except* the
  bias–variance handling, which bagging changes.
- **Imbalanced data:** still biased to majority class → `class_weight`,
  resampling, or `BalancedRandomForestClassifier` (imblearn).
- **Cannot extrapolate** (regression) — predictions bounded by training targets.

### Disadvantages
1. Training slows with **high dimensionality** (many features per split search)
2. **Less interpretable** than one tree — hundreds of trees ≈ black box
   (use feature importance / SHAP for explanations)
3. Larger memory footprint and higher prediction latency than a single tree
4. Usually beaten on accuracy by well-tuned gradient boosting on tabular data

---

## 9. RF vs Single DT vs GBDT

| | Single DT | Random Forest | GBDT |
|---|---|---|---|
| Ensemble type | — | Bagging + column sampling | Boosting |
| Base tree depth | tuned (5–10) | deep/full | shallow (3–8) |
| Reduces | — | Variance | Bias |
| Training | fast | parallel | sequential (stages) |
| Overfitting w/ more models | n/a | No (plateaus) | Yes (tune M, lr) |
| Tuning effort | low | **low (robust defaults)** | high |
| Typical accuracy (tabular) | baseline | strong | strongest |

---

## 10. Frequently Asked Interview Questions

1. What two kinds of sampling does RF do, and why is column sampling essential
   on top of bagging? (Decorrelation story.)
2. Why are RF base trees grown fully instead of pruned?
3. Show that ≈63.2% of points are in-bag. What is the OOB score and why is it a
   valid CV substitute?
4. Does adding more trees overfit an RF? Why not? What *is* the hyperparameter
   story for RF vs GBDT?
5. Bias and variance of the forest vs the base trees — what changes, what doesn't?
6. Why is RF training trivially parallelizable but boosting isn't?
7. max_features: what are the defaults for classification vs regression, and
   what happens as you lower it?
8. 🔍 (Occasional) What are Extremely Randomized Trees and when would you prefer them?
9. How is feature importance computed in RF and what are its pitfalls?
10. Train/test complexity of RF; how does it behave in very high dimensions?
11. Why might a tuned XGBoost beat RF on tabular data — and when would you still
    pick RF? (Low tuning budget, parallel training, robustness.)
12. Can RF extrapolate in regression? Why not?
