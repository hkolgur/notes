# Decision Trees — Interview Notes

## 1. What is a Decision Tree?

A **decision tree** is a supervised, non-parametric model for classification and
regression. It is a set of nested **if-else conditions** arranged as a tree:

- **Root / internal nodes** → a condition on one feature (e.g. `age < 30?`)
- **Branches** → outcomes of the condition
- **Leaf nodes** → the prediction (majority class for classification, mean/median value for regression)

Geometrically, a decision tree partitions the feature space into
**axis-parallel hyper-rectangles** (each split is a cut perpendicular to one
feature axis) and assigns one label per rectangle.

Key properties:
- **Non-parametric** — no assumption about data distribution
- **Non-linear** decision boundaries (built from many axis-parallel pieces)
- Highly **interpretable** — you can read the rules directly (a "white-box" model)

---

## 2. Impurity Measures — How Do We Choose a Split?

The core idea: at each node, pick the feature (and threshold) whose split makes
the child nodes as **pure** as possible (dominated by a single class).

### Entropy (measure of disorder / uncertainty)
For a node with class probabilities p1, p2, …, pk:

```
H(Y) = − Σ(i=1 to k) pi · log2(pi)
```

- Pure node (all one class): H = 0 (minimum)
- 50–50 binary split: H = 1 bit (maximum for 2 classes)
- Entropy curve is **concave**, peaks at uniform distribution

### Gini Impurity
```
Gini(Y) = 1 − Σ(i=1 to k) pi²
```

- Pure node: Gini = 0; 50–50 binary: Gini = 0.5 (max for 2 classes)
- Same shape/behavior as entropy but **no log computation → faster**
- This is why **CART / sklearn use Gini by default**

> **Interview point:** Gini vs entropy almost never changes the final tree much.
> Gini is preferred for speed; entropy is slightly more sensitive to class
> probabilities. Know both formulas cold.

### Information Gain (IG)
Reduction in entropy achieved by a split:

```
IG(Y, split) = H(parent) − Σ (n_child / n_parent) · H(child)
```

i.e. parent impurity minus the **weighted average** of child impurities.
Choose the split with **maximum IG** (or maximum Gini decrease).

### Misclassification Error (rarely used for splitting)
```
Error = 1 − max(pi)
```
Not sensitive enough (flat regions) — fine for evaluating, poor for growing trees.

### Entropy of Distributions (intuition)
The more **peaked** a distribution (values concentrated in a narrow range),
the **lower** its entropy; a uniform/spread-out distribution has maximum entropy.

### Mini Gini example — what a "perfect split" looks like
Node with 9 points (5 bad, 4 good). A binary feature f1 sends all 5 bad to branch 0
and all 4 good to branch 1:
```
Gini(branch 0) = 1 − (5/5)² − 0² = 0        (pure)
Gini(branch 1) = 1 − 0² − (4/4)² = 0        (pure)
Weighted child impurity = (5/9)·0 + (4/9)·0 = 0
```
⚠️ The **0 is the weighted child impurity, not the gain**. The gain here is
*maximal*: Gain = Gini(parent) − 0 = 1 − (5/9)² − (4/9)² ≈ 0.494. A feature that
drives child impurity to zero is the **best** possible split.

---

## 3. Worked Example — Play Tennis (computing IG by hand)

Dataset: 14 days, target = Play Tennis (9 Yes, 5 No).

### Step 1 — Entropy of the parent node
```
H(S) = −(9/14)·log2(9/14) − (5/14)·log2(5/14)
     = −0.643·(−0.637) − 0.357·(−1.485)
     ≈ 0.41 + 0.53 = 0.94 bits
```

### Step 2 — Split on "Wind" (Weak: 8 days → 6 Yes/2 No; Strong: 6 days → 3 Yes/3 No)
```
H(Weak)   = −(6/8)log2(6/8) − (2/8)log2(2/8) ≈ 0.811
H(Strong) = −(3/6)log2(3/6) − (3/6)log2(3/6) = 1.0

IG(Wind) = 0.94 − [ (8/14)(0.811) + (6/14)(1.0) ]
         = 0.94 − [0.463 + 0.429] = 0.048
```

### Step 3 — Split on "Outlook" (Sunny: 5 → 2Y/3N; Overcast: 4 → 4Y/0N; Rain: 5 → 3Y/2N)
```
H(Sunny) = 0.971,  H(Overcast) = 0 (pure!),  H(Rain) = 0.971

IG(Outlook) = 0.94 − [ (5/14)(0.971) + (4/14)(0) + (5/14)(0.971) ]
            = 0.94 − 0.694 = 0.246
```

### Step 4 — Decide (full comparison across all 4 features)
```
IG(Outlook)     = 0.247   ← maximum, split here
IG(Humidity)    = 0.152
IG(Wind)        = 0.048
IG(Temperature) = 0.029
```
IG(Outlook) is the largest → **split on Outlook first**.
The Overcast branch is already pure → becomes a leaf ("Yes").
Recurse on the Sunny and Rain subsets with the remaining features.

**Stopping:** recursion stops when a node is pure, features are exhausted,
max depth is reached, or the node has too few samples.

---

## 4. Handling Numerical Features

For a numerical feature, candidate thresholds are found by:
1. **Sort** the feature values — O(n log n)
2. Evaluate splits of the form `xj ≤ τ` at midpoints between consecutive
   (sorted) values — in practice only where the **class label changes**
3. Pick the threshold with maximum IG / Gini decrease

This makes training cost roughly **O(n · log n · d)** for n points, d features.

> Decision trees need **no feature scaling/normalization** — splits depend only
> on order/thresholds, not magnitudes. (Common interview point.)

### Categorical features with many values — caution
Information Gain is **biased toward features with many distinct values**
(e.g. a "date" or "ID" column gives perfectly pure but useless splits).
Fixes:
- **Gain Ratio** (C4.5): IG / SplitInfo, penalizes many-valued splits
- Convert high-cardinality categoricals to numeric via **response/probability
  encoding**: replace each category c with P(y=1 | c) computed from training data
  (e.g. zip code → fraud rate of that zip). The tree can then split on the numeric
  probability (`p < 0.2`), avoiding one-hot explosion. (Beware target leakage —
  compute encodings out-of-fold.)

---

## 5. Algorithms: What You Actually Need to Know

- **CART** is what sklearn implements: **binary splits only**, Gini for
  classification, MSE for regression, supports both feature types.
- **Gain ratio** (from C4.5) = IG / SplitInfo — the fix for IG's bias toward
  high-cardinality features (Section 4).
- Full ID3/C4.5/CART comparison: see Appendix (🔍 rarely asked in this detail).

---

## 6. Overfitting, Underfitting & Depth (bias–variance)

**Depth is the key hyperparameter** (analogous to K in KNN and α in NB):

| Depth | Behavior | Bias/Variance |
|---|---|---|
| Very deep (grow until pure leaves) | Memorizes training data; leaves with 1–2 points; noise captured; train accuracy ~100%, test poor. Small change in data → very different tree | **High variance → Overfitting** |
| Very shallow (e.g. depth 1–2, a "decision stump") | Too coarse; underlying pattern missed; predictions dominated by majority class | **High bias → Underfitting** |

Tune `max_depth` (and friends) with **cross-validation**.

### Regularization knobs (sklearn names)
- `max_depth` — cap tree depth
- `min_samples_split` — min points required to split a node
- `min_samples_leaf` — min points required in each leaf
- `max_features` — features considered per split
- `splitter` — `best` (evaluate all features, default) vs `random` (random subset — faster, more variance)
- `min_impurity_decrease` — split only if impurity drops enough
- `ccp_alpha` — **cost-complexity (post-)pruning** strength

### Pruning
- **Pre-pruning (early stopping):** stop growth via the constraints above
- **Post-pruning:** grow the full tree, then cut back branches whose removal
  doesn't hurt validation performance. Interview-level answer stops here.
  🔍 *Deep-dive:* cost-complexity pruning minimizes `Error + α·(number of leaves)`
  (`ccp_alpha` in sklearn) — formula rarely asked.

> Interview phrasing: an unpruned decision tree is a **low-bias, high-variance**
> model. Bagging many such trees (Random Forest) attacks the variance;
> boosting shallow trees (GBDT) attacks the bias.

---

## 7. Regression Trees

Same recursive splitting, but:
- **Split criterion:** minimize weighted **MSE** (variance) of children:
  `Σ (yi − ȳ_node)²` — the split that best reduces variance wins
  (MAE is an alternative)
- **Leaf prediction:** **mean** of target values in the leaf (median if using MAE/MAD —
  median absolute deviation criterion — which is more robust to outliers)
- Predictions are **piecewise-constant** → trees cannot extrapolate beyond the
  range of training targets (**classic interview trap**)

---

## 8. Time & Space Complexity

| Phase | Cost |
|---|---|
| Train | ~O(n · log n · d) (sorting each feature at each level) |
| Test | **O(depth)** per query — just follow the path (typically O(log n) for a balanced tree) |
| Space (model) | O(number of nodes) — small; store only the rules |

- Extremely fast at prediction → good for **low-latency** applications
- Training data can be discarded after training (unlike KNN)

---

## 9. Practical Considerations

### Feature scaling
Not needed — splits are threshold-based (monotonic transformations of a feature
don't change the tree).

### Multi-class classification
Supported **natively** — entropy and Gini are defined over any number of classes;
no one-vs-rest machinery needed (unlike SVM / logistic regression).

### Sweet spot
DTs shine for **large n, small-to-medium d, low-latency** requirements — training
scales O(n log n · d) but prediction is just O(depth); practical depth ≈ 5–10.

### Outliers
Largely **robust for classification** (an outlier just falls in some rectangle;
thresholds depend on order, not magnitude). In **regression trees**, outliers can
distort leaf means and split choices (MSE is sensitive) — consider MAE.
Very deep trees can still carve tiny boxes around outliers → prune/limit depth.

### Missing values
- Simple: impute, or treat "missing" as its own category
- C4.5: distribute the sample **fractionally** down all branches
- CART: **surrogate splits** (backup features that mimic the primary split) 🔍 *Deep-dive — rarely asked*
- Modern GBDT libraries (XGBoost/LightGBM) learn a **default direction** for missing values

### Imbalanced data
Impurity measures get dominated by the majority class → tree predicts majority
almost everywhere. Fixes: up/down-sampling, `class_weight='balanced'`
(weighted impurity), or threshold tuning on predicted probabilities.

### High-dimensional data
Works, but training slows (evaluate d features per node) and axis-parallel
splits struggle with sparse one-hot text data → **NB / linear models usually beat
trees on bag-of-words text**.

### Feature importance
Sum of impurity decreases contributed by each feature across the tree
(weighted by node size) → `feature_importances_`. Caveats: biased toward
high-cardinality features; correlated features split the credit. Alternative:
**permutation importance**.

### Interpretability
One of the most interpretable ML models — each prediction is a readable path of
if-else rules (valuable in medicine, credit scoring, compliance).
Interpretability drops as depth grows (and vanishes in forests/ensembles).

### Decision boundary geometry
Axis-parallel rectangles → trees need many splits to approximate a **diagonal**
linear boundary (staircase effect). A rotated dataset can be much harder —
trees are **not rotation-invariant**.

### Stability
Trees are **unstable**: small data perturbations can change the first split and
cascade into a completely different tree (high variance). This instability is
exactly why averaging many trees (Random Forest) helps so much.

---

## 10. Best & Worst Cases

**Works well when:**
1. You need interpretability / explicit decision rules
2. Mixed feature types (numeric + categorical), no scaling wanted
3. Non-linear interactions between features matter
4. Fast, low-latency predictions are required
5. As the **base learner for ensembles** (Random Forest, XGBoost, LightGBM) —
   the dominant use of trees today

**Weaknesses:**
1. Overfits easily without depth limits/pruning (high variance, unstable)
2. Axis-parallel splits → awkward for linear/diagonal boundaries
3. Cannot extrapolate (regression)
4. Greedy algorithm — each split is locally optimal; finding the globally optimal
   tree is **NP-complete**, so the final tree may be suboptimal
5. Biased splits toward high-cardinality categorical features (use gain ratio)
6. Predicted probabilities are coarse/poorly calibrated (leaf frequencies)
7. **Poor at modeling linear/additive relationships** — a simple linear regression
   can beat a tree on data with strong linear trends (e.g. time series): the tree
   approximates the line with a staircase and cannot extrapolate the trend

---

## 11. Quick Comparison

| | Decision Tree | Naive Bayes | KNN | Logistic Regression |
|---|---|---|---|---|
| Type | Rule-based, non-parametric | Generative, probabilistic | Instance-based | Discriminative, linear |
| Decision boundary | Axis-parallel rectangles | Linear (in log space) | Arbitrary/local | Linear |
| Feature scaling needed | No | No | **Yes** | Yes (for regularization/GD) |
| Train time | O(n log n · d) | O(n·d) | ~0 (lazy) | Iterative |
| Test time | O(depth) | O(d·c) | O(n·d) | O(d) |
| Interpretability | ★★★ (rules) | ★★ (word probabilities) | ★ | ★★ (weights) |
| Key hyperparameter | max_depth (+ pruning) | α | K | λ / C |
| Handles feature interactions | ✔ natively | ✘ (assumes independence) | ✔ implicitly | ✘ (unless engineered) |

---

## 12. Frequently Asked Interview Questions

1. Define entropy, Gini impurity, and information gain; write the formulas.
   When is entropy 0 and when is it maximal?
2. Gini vs entropy — do they produce different trees? Which does sklearn use and why?
3. Walk through one split selection by hand (like the Play Tennis example).
4. How are numerical features split? What's the training complexity and why?
5. Why don't decision trees need feature scaling?
6. Why is information gain biased toward high-cardinality features and what is
   gain ratio?
7. How do trees overfit and how do you prevent it (pre- vs post-pruning,
   min_samples_leaf, ccp_alpha)? Relate depth to bias–variance.
8. How does a regression tree choose splits and predict? Why can't it extrapolate?
9. ID3 vs C4.5 vs CART — key differences.
10. How do trees handle missing values (surrogate splits, default direction)?
11. What do trees' decision boundaries look like geometrically? Why do they
    struggle with diagonal boundaries?
12. Why are decision trees unstable, and how do Random Forests exploit this?
    (Sets up the ensemble follow-up: bagging reduces variance, boosting reduces bias.)
13. How is feature importance computed in a tree, and what are its pitfalls?
14. Is finding the optimal decision tree tractable? (No — NP-complete; greedy
    heuristics are used.)
15. Train/test time complexity of a decision tree vs KNN vs NB.
16. You built a decision tree on time-series data and a simple regression model
    beat it. Why can this happen? (Trees capture non-linear interactions but map
    linear relationships poorly and can't extrapolate trends.)
17. 🔍 (Rare) KS statistic vs KL divergence — which is differentiable and why does
    that matter for use as a loss function? (See Appendix.)
18. A split produces two pure children — what are the weighted child impurity and
    the information gain? (0 and maximal, respectively — don't confuse them.)

---

## Appendix — 🔍 Deep-Dive / Rarely-Asked Material (skip on first review)

### KL Divergence vs KS Statistic
- **KS statistic** = the maximum distance between two CDFs, sup |F1(x) − F2(x)|.
  Involves a supremum → **not differentiable** → can't be used as a loss in optimization.
- **KL divergence (relative entropy)** = Σ p(x)·log(p(x)/q(x)) (integral for
  continuous) — 0 when the distributions are identical, grows as they diverge.
  **Differentiable**, so usable as a loss (e.g. cross-entropy training). Called
  *relative* entropy because it matches the entropy formula but against a
  reference distribution.
- ⚠️ Don't conflate them: max CDF gap = **KS**; expected log-ratio = **KL**.
- Context where these DO come up: KL → deep learning/VAEs/cross-entropy;
  KS → data-drift detection. Not decision-tree rounds.

### Full ID3 vs C4.5 vs CART table
| | ID3 | C4.5 | CART |
|---|---|---|---|
| Split criterion | Information Gain (entropy) | Gain Ratio | Gini (classif.), MSE (regr.) |
| Feature types | Categorical only | Categorical + numeric | Categorical + numeric |
| Splits | Multi-way | Multi-way | **Binary only** |
| Regression | No | No | **Yes** |
| Pruning | None | Error-based pruning | Cost-complexity pruning |
| Used in sklearn? | — | — | ✔ (`DecisionTreeClassifier/Regressor`) |
