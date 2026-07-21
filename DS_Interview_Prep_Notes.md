# Data Scientist (Mid-Level) — Interview Prep Notes

> Format: Concept → Plain-English definition → What to say in interview → Common follow-up traps

---

## 1. Statistics & Probability

### Descriptive vs Inferential Statistics
- **Descriptive**: summarizing data you have (mean, median, std dev, distributions).
- **Inferential**: using a sample to draw conclusions about a population (hypothesis tests, confidence intervals).
- **Say**: "Descriptive tells you what happened in your data; inferential lets you generalize beyond it, with quantified uncertainty."

### Mean vs Median vs Mode
- Mean is sensitive to outliers; median is robust; mode is for categorical/most-frequent value.
- **Follow-up trap**: "When would you prefer median over mean?" → Skewed data (income, house prices).

### Variance & Standard Deviation
- Variance = average squared deviation from mean. Std dev = √variance (same units as data).
- **Say**: "Std dev gives interpretable spread in original units; variance is used mathematically because it's additive for independent variables."

### Covariance vs Correlation
- Covariance: direction of joint variability (unbounded, scale-dependent).
- Correlation: normalized covariance, bounded [-1, 1], scale-independent.
- **Say**: "Correlation lets me compare relationships across variables with different units."

### Probability Distributions (know these cold)
| Distribution | Use case | Key property |
|---|---|---|
| Normal | Natural continuous phenomena, CLT | Symmetric, defined by μ, σ |
| Binomial | # successes in n trials | Discrete, fixed n, p |
| Poisson | Count of rare events in fixed interval | λ = mean = variance |
| Exponential | Time between Poisson events | Memoryless |
| Uniform | Equal likelihood over range | Flat |

### Central Limit Theorem (CLT)
- Sample means of a sufficiently large sample (n≥30 rule of thumb) approximate a normal distribution, regardless of the population's original distribution.
- **Say**: "CLT is why we can use normal-based confidence intervals and z-tests even when the underlying data isn't normal."

### Law of Large Numbers vs CLT
- LLN: sample mean converges to true mean as n→∞.
- CLT: describes the *shape* of the sampling distribution of that mean.

### p-value
- Probability of observing data at least as extreme as what you got, **assuming the null hypothesis is true**.
- **Common misconception to avoid**: p-value is NOT the probability the null hypothesis is true.
- **Say**: "A p-value < α (e.g., 0.05) means the observed effect is unlikely under the null, so we reject it — but it doesn't quantify effect size or practical significance."

### Type I vs Type II Error
- Type I (α): false positive — rejecting a true null.
- Type II (β): false negative — failing to reject a false null.
- Power = 1 - β = probability of correctly detecting a true effect.
- **Say**: "There's a tradeoff — lowering α (fewer false positives) typically increases β (more false negatives) unless you increase sample size."

### Confidence Intervals
- Range that would contain the true parameter in X% of repeated samples (NOT "95% probability true value is in this interval" — subtle but important frequentist distinction).

### Bayesian vs Frequentist
- Frequentist: parameters are fixed, data is random; probability = long-run frequency.
- Bayesian: parameters are random variables with a prior; probability = degree of belief, updated via Bayes' theorem.
- **Say**: "Bayesian methods let you incorporate prior knowledge and get a full posterior distribution, useful with small data; frequentist is simpler and more standard for large-sample hypothesis testing."

### Bayes' Theorem
```
P(A|B) = P(B|A) * P(A) / P(B)
```
- Classic interview question: medical testing / false positive paradox (base rate fallacy).

### Central Concepts to Rehearse Out Loud
- Explain **selection bias**, **survivorship bias**, **confounding variables**, **Simpson's paradox** with a concrete example each.

---

## 2. Machine Learning Fundamentals

### Supervised vs Unsupervised vs Semi-supervised vs Reinforcement Learning
- Supervised: labeled data, predict output (regression/classification).
- Unsupervised: no labels, find structure (clustering, dimensionality reduction).
- Semi-supervised: small labeled + large unlabeled set.
- Reinforcement: agent learns via reward signals from environment interaction.

### Bias-Variance Tradeoff
- **Bias**: error from overly simplistic assumptions (underfitting).
- **Variance**: error from sensitivity to training data fluctuations (overfitting).
- Total error = Bias² + Variance + Irreducible error.
- **Say**: "A high-bias model underfits — it's too simple to capture patterns. A high-variance model overfits — it memorizes noise. The goal is the sweet spot that minimizes total generalization error."

### Overfitting vs Underfitting — How to Detect & Fix
| Symptom | Overfitting | Underfitting |
|---|---|---|
| Train error | Low | High |
| Test error | High | High |
| Fix | Regularization, more data, simpler model, dropout, early stopping | More features, more complex model, less regularization |

### Regularization
- **L1 (Lasso)**: adds `λ·Σ|w|` penalty → drives some weights to exactly 0 → feature selection.
- **L2 (Ridge)**: adds `λ·Σw²` penalty → shrinks weights smoothly, keeps all features.
- **Elastic Net**: combination of both.
- **Say**: "L1 is useful when I suspect many features are irrelevant and want sparsity; L2 when I want to control multicollinearity without eliminating features."

### Cross-Validation
- **k-fold**: split data into k parts, train on k-1, validate on remainder, rotate.
- **Stratified k-fold**: preserves class distribution — important for imbalanced classification.
- **Time-series split**: no shuffling; respects temporal order (never train on future data).
- **Say**: "I pick stratified k-fold for imbalanced classification, and a rolling/time-based split for any temporal data to avoid leakage."

### Data Leakage
- When information from outside the training set (often from the future, or the target itself) leaks into features.
- **Classic example**: normalizing/scaling using the full dataset (including test set) before splitting.
- **Say**: "I always fit preprocessing (scalers, encoders, imputers) only on the training fold, then transform validation/test with those same fitted parameters."

---

## 3. Core Algorithms — Be Ready to Explain Intuition + Math + Pros/Cons

### Linear Regression
- Assumes linear relationship, models `y = Xβ + ε`.
- Assumptions: linearity, independence, homoscedasticity (constant variance of residuals), normality of residuals, no multicollinearity.
- **Follow-up**: "How do you detect multicollinearity?" → VIF (Variance Inflation Factor), correlation matrix.

### Logistic Regression
- Models log-odds as linear combination of features; output passed through sigmoid.
- Loss: **log-loss / cross-entropy**, not MSE (MSE would be non-convex for classification).
- **Say**: "Logistic regression is interpretable via odds ratios and is a strong baseline before trying more complex models."

### Decision Trees
- Splits data by feature thresholds to maximize purity (Gini impurity or entropy/information gain).
- Prone to overfitting if unconstrained (depth, min samples per leaf).
- **Gini vs Entropy**: both measure impurity; Gini is computationally cheaper, entropy is more sensitive to class distribution changes. In practice results are similar.

### Random Forest
- Ensemble of decision trees using **bagging** (bootstrap aggregating) + random feature subsets per split.
- Reduces variance vs a single tree; robust to overfitting; less interpretable.
- **Say**: "Each tree sees a bootstrapped sample and a random subset of features, which decorrelates the trees so averaging reduces variance without much bias increase."

### Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Builds trees **sequentially**, each new tree fits the residual errors (gradient of the loss) of the ensemble so far.
- Key hyperparameters: learning rate, number of estimators, max depth, subsample, regularization (λ, α).
- **Bagging vs Boosting**: bagging reduces variance (parallel, independent models); boosting reduces bias (sequential, focuses on hard examples) but can overfit if unregularized.

### Support Vector Machines (SVM)
- Finds the hyperplane that maximizes margin between classes.
- **Kernel trick**: implicitly maps data to higher dimensions (RBF, polynomial) to handle non-linear boundaries without explicitly computing the transformation.
- Sensitive to feature scaling; less common now vs. gradient boosting/deep learning for tabular data, but good to know conceptually.

### K-Nearest Neighbors (KNN)
- Lazy learner — no training phase, classifies by majority vote of k nearest points.
- Sensitive to feature scaling and curse of dimensionality.
- Choosing k: small k → high variance/noisy; large k → high bias/smoother boundary.

### Naive Bayes
- Assumes feature independence given the class (rarely true, but works well in practice — e.g., spam filtering, text classification).
- Fast, works well with high-dimensional sparse data (like bag-of-words).

### K-Means Clustering
- Partitions data into k clusters minimizing within-cluster sum of squares.
- Requires choosing k in advance (elbow method, silhouette score).
- Sensitive to initialization (use k-means++) and outliers; assumes spherical, similarly-sized clusters.

### Hierarchical Clustering
- Builds a dendrogram; no need to pre-specify k. Agglomerative (bottom-up) is most common.

### DBSCAN
- Density-based clustering; finds arbitrarily shaped clusters and naturally identifies outliers/noise. Doesn't need k, but needs eps/min_samples tuning.

### PCA (Principal Component Analysis)
- Linear dimensionality reduction; finds orthogonal directions (principal components) that maximize variance explained.
- **Say**: "PCA is unsupervised and finds directions of maximum variance — useful for reducing dimensionality/noise and visualization, but components lose direct interpretability, and it assumes linear relationships."
- Always standardize features before PCA (scale-sensitive).

---

## 4. Model Evaluation

### Classification Metrics
| Metric | Formula | When it matters |
|---|---|---|
| Accuracy | (TP+TN)/Total | Balanced classes only |
| Precision | TP/(TP+FP) | Cost of false positives is high (e.g., spam) |
| Recall (Sensitivity) | TP/(TP+FN) | Cost of false negatives is high (e.g., cancer detection, fraud) |
| F1 Score | 2·(P·R)/(P+R) | Balance precision & recall, imbalanced classes |
| ROC-AUC | Area under TPR vs FPR curve | Ranking quality, threshold-independent |
| PR-AUC | Area under Precision-Recall curve | Better than ROC-AUC for **imbalanced** data |

- **Say when asked "accuracy isn't enough"**: "With 99% class imbalance, a model predicting the majority class always gets 99% accuracy but is useless — I'd use precision/recall/F1 or PR-AUC instead."

### Regression Metrics
- **MAE**: average absolute error, robust to outliers, interpretable in original units.
- **MSE/RMSE**: penalizes large errors more (squared term) — sensitive to outliers.
- **R²**: proportion of variance explained; can be misleading with non-linear relationships or overfitting (adjusted R² penalizes extra features).

### Confusion Matrix
- Always be ready to draw this out and derive precision/recall/F1 from it live.

### ROC Curve vs Precision-Recall Curve
- ROC-AUC can look overly optimistic on imbalanced datasets (because TN is huge) — PR curve focuses on the positive class performance, which is usually what matters in imbalanced problems (fraud, disease detection).

---

## 5. Feature Engineering & Data Preprocessing

### Handling Missing Data
- Understand **mechanism**: MCAR (missing completely at random), MAR (missing at random, depends on observed data), MNAR (missing not at random, depends on unobserved value itself).
- Techniques: mean/median/mode imputation, KNN imputation, model-based imputation (e.g., MICE), or a missing-indicator flag.
- **Say**: "The right method depends on why data is missing — for MNAR, imputation can introduce bias, so I'd flag missingness as its own feature rather than blindly impute."

### Handling Outliers
- Detect via IQR (1.5×IQR rule), z-score, isolation forest.
- Treat via capping/winsorizing, transformation (log), or removal — depends on whether it's a data error or a genuine rare event.

### Categorical Encoding
- **One-hot encoding**: for low-cardinality nominal features (no ordinal relationship).
- **Label/ordinal encoding**: for ordinal categories.
- **Target/mean encoding**: for high-cardinality features — but risk of leakage, needs proper CV-based encoding.
- **Embeddings**: for very high-cardinality categorical features in deep learning.

### Feature Scaling
- **Standardization (z-score)**: needed for distance-based models (KNN, SVM, PCA) and gradient-based optimization (neural nets, logistic regression).
- **Min-Max scaling**: bounds to [0,1], sensitive to outliers.
- **Tree-based models (RF, GBM)**: scale-invariant, don't need scaling.

### Class Imbalance Techniques
- Resampling: SMOTE (synthetic oversampling), random undersampling/oversampling.
- Algorithmic: class weights, focal loss.
- Evaluation: use PR-AUC/F1 instead of accuracy.
- **Say**: "I'd try class weighting first since it's simpler and doesn't distort the data distribution; SMOTE if that's insufficient."

---

## 6. Deep Learning Basics (know the fundamentals even if not a DL-heavy role)

### Neural Network Building Blocks
- Weights, biases, activation functions (ReLU, sigmoid, tanh, softmax), forward pass, backpropagation (chain rule to compute gradients), gradient descent variants (SGD, Adam, RMSprop).

### Why ReLU over Sigmoid in hidden layers
- Avoids vanishing gradient problem; computationally cheap; sparse activation.

### Vanishing / Exploding Gradients
- Occurs in deep networks with certain activations/initializations; addressed via ReLU, batch normalization, residual connections, gradient clipping, careful weight init (Xavier/He).

### Regularization in DL
- Dropout (randomly zero neurons during training), batch normalization, early stopping, weight decay (L2), data augmentation.

### CNN vs RNN vs Transformer (high level)
- **CNN**: spatial locality — images, exploits translation invariance via convolution/pooling.
- **RNN/LSTM/GRU**: sequential data, maintains hidden state across time steps, but struggles with long-range dependencies (LSTM/GRU mitigate via gates).
- **Transformer**: attention mechanism lets every token attend to every other token in parallel — captures long-range dependencies better, parallelizable (unlike RNNs), backbone of modern LLMs.

### Batch Size & Learning Rate
- Small batch: noisier gradient estimates, can generalize better, slower per epoch.
- Learning rate: too high → diverges/oscillates; too low → slow convergence/stuck in local minima. Learning rate schedules/warm-up are common fixes.

---

## 7. SQL — Practice These Patterns

### Core clauses order (execution order, not written order)
`FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`

### Joins
- INNER, LEFT/RIGHT, FULL OUTER, CROSS, SELF JOIN — know when each applies and be ready to draw Venn diagrams.

### Window Functions (very commonly tested)
```sql
SELECT
  employee_id,
  department,
  salary,
  RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank,
  AVG(salary) OVER (PARTITION BY department) AS dept_avg_salary
FROM employees;
```
- Know difference: `ROW_NUMBER()` (unique sequential), `RANK()` (ties share rank, gaps after), `DENSE_RANK()` (ties share rank, no gaps).

### GROUP BY vs Window Functions
- GROUP BY collapses rows; window functions keep all rows while adding aggregated context — key distinction interviewers probe.

### Common query patterns to rehearse
- Second highest / Nth highest value (subquery, `LIMIT/OFFSET`, or `DENSE_RANK`).
- Running totals / moving averages (`SUM() OVER (ORDER BY ... ROWS BETWEEN ...)`).
- Finding duplicates (`GROUP BY ... HAVING COUNT(*) > 1`).
- Cohort/retention analysis using self-joins or window functions on date differences.

---

## 8. Python / Coding Round Tips

- Be fluent in **pandas** (groupby, merge, pivot_table, apply/vectorization, handling NaNs) and **numpy** (vectorized operations vs loops).
- Know **time/space complexity (Big-O)** of common operations — e.g., `list` lookup O(n) vs `dict`/`set` O(1).
- Practice: array/string manipulation, two-pointer, sliding window, hashmap-based frequency counting, basic recursion — these show up even in "DS" interviews as warm-ups.
- Be ready to write a function from scratch: e.g., implement k-means, compute precision/recall from a confusion matrix, or write a SQL-equivalent groupby in pandas.
- **Say when writing code**: narrate your thought process, mention time/space complexity, and test edge cases (empty input, single element, duplicates, nulls) out loud.

---

## 9. A/B Testing & Experimentation

### Designing an A/B Test — Structure Your Answer
1. **Define the hypothesis & metric** (primary KPI, guardrail metrics).
2. **Determine sample size / power analysis** (effect size, baseline conversion rate, significance level, power — usually 80%).
3. **Randomization** — ensure proper unit of randomization (user vs session) to avoid interference/leakage.
4. **Run the test** — avoid peeking early (multiple testing / p-hacking risk).
5. **Analyze** — statistical test (t-test/z-test for means, chi-square for proportions), check for novelty effects, Simpson's paradox across segments.
6. **Decide** — statistical AND practical significance.

### Common Pitfalls to Mention
- **Peeking problem**: checking results repeatedly inflates false positive rate — use sequential testing methods (e.g., always-valid p-values) if early stopping is needed.
- **Novelty/primacy effects**: initial reaction to a change may not reflect long-term behavior.
- **Network effects/interference**: users' outcomes affect each other — violates SUTVA assumption; consider cluster/geo-based randomization.
- **Multiple comparisons**: testing many metrics increases false discovery — apply Bonferroni or Benjamini-Hochberg correction.

---

## 10. ML System Design / Case Study Framework

Use this as a checklist for open-ended "build a model to predict X" or "design a recommendation system" questions:

1. **Clarify the problem**: business objective, who uses the output, what decision it drives, latency/scale constraints.
2. **Define success metric**: business metric (revenue, retention) vs model metric (AUC, RMSE) — explain the link.
3. **Data**: what's available, how it's labeled, potential biases/leakage, how much history.
4. **Feature engineering**: what signals would be predictive; discuss real-time vs batch feature availability.
5. **Model choice**: start simple (baseline: logistic regression/heuristic), justify moving to complex models (GBM, deep learning) based on data size/nonlinearity needs.
6. **Evaluation**: offline metrics + how you'd validate with an online A/B test.
7. **Deployment considerations**: latency, retraining cadence, monitoring for data/model drift, fallback/rollback strategy.
8. **Ethical/fairness considerations**: bias across subgroups, if relevant to the product.

- **Say**: "I always start by clarifying the business objective before jumping to models — the right metric and constraints shape everything downstream."

---

## 11. Behavioral / Communication Tips

- Use **STAR** (Situation, Task, Action, Result) for behavioral questions — always end with a quantified result/impact.
- Prepare 2-3 project deep-dives where you can discuss: the business problem, why you chose your approach over alternatives, a mistake/setback and what you learned, and the measurable outcome.
- When explaining technical concepts, **lead with intuition before math** — interviewers weigh communication clarity heavily for mid-level roles (you're expected to explain results to non-technical stakeholders).
- Prepare thoughtful questions for the interviewer (team structure, model deployment process, how success is measured) — signals seniority and genuine interest.

---

## Quick Self-Check Before the Interview
- [ ] Can I explain bias-variance tradeoff in under 30 seconds with an example?
- [ ] Can I derive precision/recall from a confusion matrix without hesitating?
- [ ] Can I write a window function query from memory?
- [ ] Can I walk through designing an A/B test end-to-end?
- [ ] Can I explain one past project's business impact in numbers?
- [ ] Do I have 2 clarifying questions ready for any open-ended case study?

# Data Science / MLE Interview Blueprint (Financial Domain Focus)

### Q1: Why must we scale tabular features (like "Income" and "Credit Score") before training a linear model or neural network using Gradient Descent?
**Answer:**
* **The Problem:** Financial data has massive scale differences. For example, "Annual Income" ranges from $20,000 to $1,000,000+, while "Credit Score" only ranges from 300 to 850. 
* **The Landscape Impact:** If unscaled, a tiny tweak to the Income weight drastically impacts the loss compared to a tweak to the Credit Score weight. This stretches the loss landscape into a narrow, elongated canyon.
* **The Optimization Impact:** The gradient vector ends up pointing sharply across the narrow canyon walls rather than down the valley floor toward the true minimum. Gradient Descent **oscillates and bounces violently**, forcing you to lower the learning rate and drastically slowing down convergence.
* **The Solution:** Scaling (Standardization/Normalization) turns the canyon into a symmetric, circular bowl. The gradient points directly at the minimum, allowing for a much higher learning rate and vastly faster training.

---

### Q2: Do Tree-Based Models (like Random Forests or XGBoost) require feature scaling? Why or why not?
**Answer:**
* **No, tree-based models are completely invariant to feature scaling.** 
* **The Reason:** Decision trees split nodes based on thresholds (e.g., `Income > 75,000`). A split is chosen strictly based on how well it separates the data classes (using Information Gain or Gini Impurity). 
* Whether Income is scaled between 0 and 1, or left in raw dollar amounts, the relative ranking of the data points remains identical. The tree will simply adjust its split threshold to match the scale, resulting in the exact same model architecture and performance.

---

### Q3: Why is Gradient Boosting (XGBoost/LightGBM) the industry standard for financial tabular data, and how does its optimization differ from a Neural Network?
**Answer:**
* **Why it rules finance:** Tabular financial data (fraud, credit risk, churn) is full of unscaled variables, missing entries, extreme outliers, and sharp non-linear boundaries—all things neural networks struggle to process without extensive data preprocessing. Trees handle these natively.
* **The Optimization Difference:** 
  * A **Neural Network** optimizes in *parameter space*. It keeps the model structure fixed and tweaks a static set of internal weights using Gradient Descent ($w \leftarrow w - \eta \nabla L$).
  * **Gradient Boosting** optimizes in *function space*. It starts with a base prediction and iteratively fixes errors by adding entirely new decision trees ($F_{t+1}(x) \leftarrow F_t(x) + \eta h_t(x)$). 
* For standard MSE loss, the negative gradient simplifies to the **residual error** ($y - F(x)$). Therefore, training a new tree to predict the current model's mistakes is mathematically equivalent to taking a step down the path of steepest descent.

---

### Q4: Financial datasets (like Credit Fraud or Loan Default) are notoriously imbalanced (e.g., 99.9% legitimate, 0.1% fraud). How does this affect Gradient Descent, and how do you fix it at the loss level?
**Answer:**
* **The Impact on GD:** If 99.9% of your dataset is legitimate transactions, the majority class will completely dominate the gradient calculations during backpropagation. The gradient vectors will align almost exclusively with directions that optimize accuracy for the legitimate class, allowing the model to simply predict "Never Fraud" and stall out.
* **The Loss-Level Fixes:**
  1. **Class Weights:** We modify the loss function to penalize misclassifications of the minority class more severely. For instance, in Binary Cross-Entropy, we multiply the loss of the fraud class by a weight factor (e.g., $W_{fraud} = 100$). This scales up the minority class gradients, forcing the optimizer to prioritize learning its features.
  2. **Focal Loss:** Frequently used in advanced models, it dynamically downweights the loss assigned to easy-to-classify examples (the massive pool of legitimate transactions) and forces the gradient steps to focus purely on hard, ambiguous examples (the fraud cases).

# Interview Prep: Troubleshooting Logistic Regression vs. XGBoost

## The Interview Question
> *"You trained an XGBoost model and a Logistic Regression model on a credit card loan dataset. The XGBoost model performs exceptionally well, but the Logistic Regression model is performing significantly worse. How do you troubleshoot the Logistic Regression model?"*

### The Ideal Structural Framework
A senior-level answer avoids listing random ideas. Instead, it breaks the problem down into three core technical buckets: **Data Assumptions**, **Feature Engineering**, and **Optimization Bugs**.

---

### 1. Address Data Assumptions (Feature Scaling)
*   **The Pitch:** First, immediately check if the features were scaled. XGBoost is invariant to feature scaling because it uses node-splitting thresholds. Logistic Regression, however, relies on gradient descent over weights.
*   **The Problem:** If financial variables like `Annual Income` (scale of millions) and `Debt-to-Income Ratio` (scale of decimals) are left unscaled, the loss landscape becomes an elongated canyon. The gradient steps will oscillate violently, preventing convergence.
*   **The Fix:** Apply a `StandardScaler` to normalize the feature variances before training.

### 2. Address Structural Form (Non-Linearities & Interactions)
*   **The Pitch:** Logistic Regression assumes a strictly linear relationship between the log-odds of the target and the features. It cannot natively capture non-linear patterns or feature interactions.
*   **The Example:** In loan risk, a high debt-to-income ratio matters much more *only if* the credit score is low. XGBoost captures this interaction natively by splitting down sequential branches. Logistic Regression misses it entirely.
*   **The Fix:** Manually engineer interaction terms (e.g., multiplying features together), apply polynomial transformations, or use feature binning on highly non-linear variables.

### 3. Address Data Outliers and Missing Values
*   **The Pitch:** Financial tabular data is notorious for extreme outliers (e.g., a multi-million dollar transaction) and missing entries. XGBoost handles missing data natively by assigning default directions to branches and is highly robust to outliers.
*   **The Problem:** Logistic Regression is heavily dragged off course by outliers because it directly factors the feature distances into its loss calculation. It also fails completely if missing values (`NaN`) are present.
*   **The Fix:** Implement an outlier capping strategy (like clipping data at the 1st and 99th percentiles) and explicitly impute missing fields using the median or a dedicated indicator column.

### 4. Address the Loss Level (Class Imbalance)
*   **The Pitch:** Because loan default is a highly imbalanced problem, standard Logistic Regression might just predict "No Default" for everything to naively optimize overall accuracy.
*   **The Fix:** Ensure that the class imbalance is handled explicitly, such as setting the `class_weight='balanced'` parameter in `scikit-learn` to artificially scale up the gradients of the minority default class during optimization.


### Case Study Cheat Sheet: XGBoost vs. Logistic Regression Gap

| Root Cause Category | Why XGBoost Succeeded | Why Logistic Regression Failed | The Actionable Fix |
| :--- | :--- | :--- | :--- |
| **Feature Scaling** | Scale-invariant; uses split thresholds. | Stalls/oscillates due to elongated loss landscapes. | Apply `StandardScaler`. |
| **Interactions/Non-Linearity** | Natively maps interactions down tree branches. | Assumes strict linearity; completely blind to feature combos. | Manually add interaction terms ($X_1 \times X_2$). |
| **Outliers & Missing Data** | Immune to outliers; handles missing values natively. | Outliers heavily warp the decision boundary; fails on missing fields. | Clip outliers at 99th percentile; impute missing values. |
| **Class Imbalance** | Can easily isolate minority clusters in leaves. | Gradients are completely dominated by the majority class. | Use `class_weight='balanced'` or adjust threshold. |

