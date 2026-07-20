# Calculus & Gradient Descent — Interview Notes

## 1. Derivatives — the Foundation

### Definition
The derivative measures the **instantaneous rate of change** of f at x — the
slope of the tangent line:

```
f'(x) = lim(h→0) [ f(x+h) − f(x) ] / h
```

Interpretation for ML: *if I nudge the input slightly, how much does the output
change, and in which direction?* This is exactly what training needs — how does
the loss change if I nudge a weight?

### Rules you must know cold
| Rule | Formula |
|---|---|
| Power | d/dx xⁿ = n·xⁿ⁻¹ |
| Constant / sum | (c·f)' = c·f'; (f+g)' = f' + g' |
| Product | (f·g)' = f'g + fg' |
| Quotient | (f/g)' = (f'g − fg') / g² |
| **Chain rule** | d/dx f(g(x)) = f'(g(x)) · g'(x) |
| Exponential / log | (eˣ)' = eˣ; (ln x)' = 1/x |

**The chain rule is the most important rule in ML** — backpropagation is nothing
but the chain rule applied repeatedly through the layers of a network.

### Common ML derivatives (memorize)
```
Sigmoid:  σ(x) = 1/(1+e⁻ˣ)      →  σ'(x) = σ(x)·(1 − σ(x))
tanh:     tanh'(x) = 1 − tanh²(x)
ReLU:     ReLU'(x) = 1 if x > 0, else 0   (undefined at 0 → use subgradient)
log-loss: d/dz [−y·log σ(z) − (1−y)·log(1−σ(z))] = σ(z) − y   (beautifully simple!)
```

### Critical points, maxima & minima
- f'(x) = 0 → **critical point** (candidate max/min/saddle)
- Local Minima:A point where the loss is lower than all its immediately surrounding points, but not the lowest point on the entire landscape (the global minimum
- Plateaus: A vast, flat or nearly flat region of the loss landscape where the gradient becomes exceptionally close to zero  in almost all directions
- Saddle Points:A point where the gradient is zero ∇L = 0, but the slope bends upward in some directions and downward in others (resembling a horse saddle)
- Second derivative test: f''(x) > 0 → local **min**; f''(x) < 0 → local **max**;
  f''(x) = 0 → inconclusive (possible saddle/inflection)
### Optimization Obstacles Summary

| Obstacle | Condition | The Problem | Key Remedies |
| :--- | :--- | :--- | :--- |
| **Local Minima** | $\nabla L = 0$<br>All directions bend UP | Traps the model in a sub-optimal bowl. (Rare in deep learning). | Mini-batch SGD noise, Learning Rate Schedulers. |
| **Saddle Points** | $\nabla L = 0$<br>Some directions UP, some DOWN | The primary trap in deep learning; stalls standard gradient descent completely. | Momentum, NAG, Adam Optimizer. |
| **Plateaus** | $\nabla L \approx 0$<br>Flat in almost all directions | Causes vanishing gradients; progress crawls to a halt. | ReLU activations, Batch Normalization, Adam. |

---

## 2. Partial Derivatives & the Gradient

### Partial derivative
For multivariable f(x, y): **∂f/∂x** = derivative w.r.t. x, treating every other
variable as a constant.

```
f(x,y) = 3x²y + y³
∂f/∂x = 6xy          ∂f/∂y = 3x² + 3y²
```

### Gradient
The **gradient** stacks all partial derivatives into a vector:

```
∇f = [ ∂f/∂x₁, ∂f/∂x₂, …, ∂f/∂xd ]ᵀ
```

**Key property (classic interview question):** ∇f points in the direction of
**steepest ascent**, and its magnitude is the rate of increase in that direction.
Therefore **−∇f is the direction of steepest descent** — the whole basis of
gradient descent.

### Why does the gradient point in the steepest-ascent direction? (full explanation)

**Picture:** you're standing on a hillside where f(x, y) = height. You can walk
in any compass direction. Which direction increases your height fastest?

**Step 1 — Directional derivative.** The slope you experience depends on which
way you walk. The slope of f in the direction of a **unit vector** u (a direction
arrow of length 1) is called the *directional derivative*, and it turns out to be
just a dot product:

```
D_u f = ∇f · u        (slope of f if you walk along direction u)
```

Sanity check: if u points purely along the x-axis, u = (1, 0), then
∇f·u = ∂f/∂x — the partial derivative, exactly as expected. The directional
derivative generalizes partials to *any* direction.

**Step 2 — Rewrite the dot product using the angle.** For any two vectors,
`a · b = |a|·|b|·cos θ`, where θ is the angle between them. Since u is a unit
vector (|u| = 1):

```
D_u f = ∇f · u = |∇f| · |u| · cos θ = |∇f| · cos θ
```

**Step 3 — Maximize over directions.** |∇f| is fixed at your current location —
the only thing you control is θ (which way you face). cos θ ranges from −1 to +1:

| You walk… | θ | cos θ | Slope you feel |
|---|---|---|---|
| **Along ∇f** | 0° | +1 | **+\|∇f\| — steepest ascent** |
| Perpendicular to ∇f | 90° | 0 | 0 — a level/contour direction |
| **Against ∇f (−∇f)** | 180° | −1 | **−\|∇f\| — steepest descent** |

That's the whole proof: the slope in direction u is |∇f|·cos θ, which is largest
when θ = 0 (walk along the gradient) and most negative when θ = 180°
(walk against it). Hence GD steps along **−∇f**.

**Concrete example.** f(x,y) = x² + y² (a bowl). At the point (3, 4):
∇f = (2x, 2y) = (6, 8), |∇f| = 10.
- Walking along (6,8)/10: slope = +10 (fastest way up the bowl)
- Walking along (−6,−8)/10: slope = −10 (fastest way down — toward the minimum
  at the origin, which is exactly where −∇f points!)
- Walking along (8,−6)/10 (perpendicular): slope = (6·8 + 8·(−6))/10 = 0 —
  you're circling the bowl at constant height (moving along a contour line).

**Bonus insight:** the gradient is always **perpendicular to contour lines**
(the θ = 90° row). This is why unscaled features hurt GD: elongated elliptical
contours make −∇f point *across* the valley instead of *along* it → zig-zagging
(see Section 3).

### Hessian — the multivariable second derivative
*(second-order concept; comes up in "why not Newton's method" and saddle-point questions)*

**Start in 1-D.** The first derivative f' tells you the *slope*; the second
derivative f'' tells you the **curvature** — how the slope itself is changing:
- f'' > 0 → curve bends **upward** (U shape, like x²) → a critical point here is a **minimum**
- f'' < 0 → curve bends **downward** (∩ shape, like −x²) → a **maximum**

**Now go multivariable.** With d variables there isn't one second derivative —
there's one for every *pair* of variables. Collect them all into a d×d matrix,
the **Hessian**:

```
H[i][j] = ∂²f / ∂xi ∂xj

For f(x, y):        H = | ∂²f/∂x²    ∂²f/∂x∂y |
                        | ∂²f/∂y∂x   ∂²f/∂y²  |
```

Diagonal entries = curvature along each axis; off-diagonals = how the slope in
one direction changes as you move in another (the "twist").

**Worked example.** f(x, y) = x² + 3y²  (an elongated bowl):
```
∂f/∂x = 2x → ∂²f/∂x² = 2       ∂²f/∂x∂y = 0
∂f/∂y = 6y → ∂²f/∂y² = 6       ∂²f/∂y∂x = 0
H = | 2  0 |
    | 0  6 |
```
Curvature is +2 along x and +6 along y — it curves *up in every direction* →
the critical point (0,0) is a minimum. Note the y-direction is 3× steeper: this
"uneven curvature" is exactly what makes GD zig-zag, and what Newton's method
(next 🔍 section) corrects for.

**Reading the Hessian at a critical point (∇f = 0):** the diagnosis is done via
the **eigenvalues** of H — intuitively, the curvatures along the surface's own
principal directions (which may be rotated relative to the axes):

| Eigenvalues of H | Shape at the point | Verdict |
|---|---|---|
| All > 0 ("**positive definite**") | Bowl — curves up in every direction | Local **minimum** |
| All < 0 (negative definite) | Dome — curves down in every direction | Local **maximum** |
| **Mixed signs** | Pringle chip / horse saddle — up in one direction, down in another | **Saddle point** |

*"Positive definite"* is just the matrix way of saying "curves upward no matter
which direction you look" (formally: vᵀHv > 0 for every direction v).

**Why interviewers care:**
1. It explains **saddle points** — ∇f = 0 yet not a minimum (mixed curvature).
   In high-dimensional deep nets, most zero-gradient points are saddles, not
   local minima (with d directions it's rare for *all* curvatures to agree).
2. It's the ingredient Newton's method uses (below).
3. **Convexity** in n-D = Hessian positive semi-definite everywhere
   (ties back to the Convexity subsection).

### Convexity (frequently asked)
- f is **convex** if the line segment between any two points on the curve lies
  on/above the curve (f'' ≥ 0 in 1-D; Hessian positive semi-definite in n-D)
- **Convex → any local minimum is the global minimum** → GD (with a suitable
  step size) converges to the global optimum
- Convex losses: linear regression MSE, logistic loss, hinge loss (SVM)
- **Non-convex:** neural network losses — many local minima and (mostly) saddle
  points; GD finds a "good enough" minimum

---

## 3. Gradient Descent — the Core Algorithm

Iteratively step **against the gradient** of the loss L(w):

```
Repeat until convergence:
    w := w − η · ∇L(w)
```

- **η = learning rate** (step size) — THE key hyperparameter
- Converged when: gradient ≈ 0, loss change < tolerance, or max iterations

### Learning rate behavior (guaranteed interview question)
| η | Behavior |
|---|---|
| Too small | Painfully slow convergence; can stall on plateaus |
| Too large | Overshoots the minimum; loss oscillates or **diverges** |
| Just right | Smooth, fast decrease |
| Common fix | **Decay schedule** — start larger, shrink over time (step decay, exponential, 1/t, cosine) |

### Why GD instead of solving analytically?
Linear regression has a closed form w = (XᵀX)⁻¹Xᵀy, but:
- Matrix inversion is **O(d³)** — infeasible for large d
- Most models (logistic regression, NNs) have **no closed form**
- GD only needs gradients → works for any differentiable loss

### Feature scaling matters for GD (link to your KNN/DT notes!)
# Why Feature Scaling Speeds Up Gradient Descent

### 1. The Core Transformation
Feature scaling transforms an **elongated, narrow canyon** loss landscape into a **perfectly round, symmetric bowl**. 
Because a small change in weight for "Size" sft changes the price drastically compared to "Bedrooms", the loss landscape gets stretched out. Instead of a round bowl, it becomes a narrow, steep canyon

### 2. The Problem (Unscaled Features)
* **Example:** Feature $X_1$ (range: 100 to 1,000,000) vs. Feature $X_2$ (range: 1 to 5).
* **Landscape:** Extreme contour distortion (highly eccentric ellipses).
* **Gradient Behavior:** The gradient vector points almost entirely toward the steep canyon walls rather than down the length of the valley toward the true minimum.
* **Consequence:** The optimization path **oscillates violently back and forth**. You are forced to use a tiny learning rate ($\eta$) to prevent divergence, making training painfully slow.

### 3. The Solution (Scaled Features)
* **Approach:** Bring all features to a similar scale (e.g., via Standardization or Min-Max Normalization).
* **Landscape:** Symmetrical, circular contours.
* **Gradient Behavior:** The gradient points **directly at the global minimum** from almost any position.
* **Consequence:** Oscillations drop to zero. You can safely use a **much larger learning rate ($\eta$)**, allowing the optimizer to take direct, massive strides straight to the bottom.

---

### Common Scaling Math Reference

#### Standard Score (Z-Score Standardization)
Centers data around $0$ with a standard deviation of $1$. Best for algorithms assuming normally distributed data.
$$x_{new} = \frac{x - \mu}{\sigma}$$
*(where $\mu$ is the mean and $\sigma$ is the standard deviation)*

#### Min-Max Normalization
Bounds data strictly between a specified range (usually 0 and 1). Best when you know your data bounds and have no massive outliers.
$$x_{new} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

---

## 4. Worked Example — GD on Simple Linear Regression (by hand)

Model: ŷ = w·x + b.  Loss: MSE = (1/n)·Σ(yi − (w·xi + b))²

### Step 1 — Partial derivatives (know this derivation!)
```
∂L/∂w = (−2/n)·Σ xi·(yi − ŷi)
∂L/∂b = (−2/n)·Σ (yi − ŷi)
```

### Step 2 — One iteration on toy data
Data: (1, 2), (2, 4), (3, 6)  (true line: y = 2x). Start w = 0, b = 0, η = 0.1.
```
Predictions: ŷ = 0, 0, 0 → errors (y − ŷ): 2, 4, 6
∂L/∂w = (−2/3)·(1·2 + 2·4 + 3·6) = (−2/3)·28 = −18.67
∂L/∂b = (−2/3)·(2 + 4 + 6)       = −8

Update:  w := 0 − 0.1·(−18.67) = 1.867
         b := 0 − 0.1·(−8)     = 0.8
```
After one step w jumped from 0 toward the true value 2. Repeating drives
(w, b) → (2, 0). Being able to run 1–2 iterations by hand is a common screen.

---

## 5. GD Variants: Batch vs Stochastic vs Mini-Batch

| | **Batch GD** | **SGD** | **Mini-Batch GD** |
|---|---|---|---|
| Gradient computed on | Entire dataset | **1 random point** | Small batch (32–512) |
| Update cost | O(n·d) per step | O(d) per step | O(B·d) per step |
| Gradient quality | Exact | Very noisy | Noisy but averaged |
| Path to minimum | Smooth | Erratic/oscillating | In between |
| Memory | Full data needed | Tiny | Batch only |
| In practice | Small data only | Rare in pure form | **Default everywhere** (GPUs parallelize a batch) |

- SGD's noise is not purely bad: it helps **escape saddle points and shallow
  local minima** in non-convex problems (classic interview point)
- SGD never fully settles — it bounces around the minimum → decay η over time
- "SGD" in papers/libraries almost always means **mini-batch** SGD
- An **epoch** = one full pass over the training data

---

## 6. Gradient Descent Optimizers (the evolution story)

**The one-sentence framing interviewers want:** how do we get a ball to roll
down a rugged, unpredictable loss surface to the minimum as fast and safely as
possible? Every optimizer below exists purely to patch a specific flaw in the
one before it — memorize the story as a *chain of complaints and fixes*, not as
six unrelated formulas.

```
[Vanilla GD] ──► slow on gentle slopes; stuck dead in flat/saddle zones
      │
[+ Momentum] ──► fast now, but overshoots the minimum and oscillates
      │            (it built up speed with no way to brake)
[+ NAG]      ──► brakes early by looking ahead — but still uses ONE global η
      │            for every parameter
      ▼
[ AdaGrad ]  ──► per-parameter η fixes that — but the denominator only
      │            grows, so the effective η decays to 0 and training freezes
      ▼
[RMSProp]    ──► keeps AdaGrad's per-parameter idea, fixes the "decays to
      │            zero and freezes" flaw with a decaying memory instead of a sum
      ▼
   [Adam]    ──► merges Momentum's velocity with RMSProp's per-parameter
                   scaling — the current default
```

### Step 1 — Momentum: give the hiker inertia (fixes: slow + gets stuck)
**Problem:** plain GD ("the blind hiker") re-evaluates the slope from scratch at
every single step. On a long gentle slope this is painfully slow; in a flat
saddle region the gradient is ≈0, so the hiker simply stops dead.

**Fix — "the rolling bowling ball":** instead of resetting your speed every
step, accumulate a **velocity** — an exponentially decaying moving average of
past gradients — and move using that velocity instead of the raw gradient:
```
v := β·v + η·∇L(w)        (β ≈ 0.9 — how much past velocity carries over)
w := w − v
```
Rolling downhill for several steps builds up speed, and that accumulated speed
is exactly what carries the ball straight through a flat saddle region where the
instantaneous gradient offers no help. This also damps the zig-zag in narrow
ravines, since the sideways components of successive gradients partially cancel
while the forward component reinforces.

**New flaw it introduces:** momentum has no brakes. Once it's built up speed
heading toward a minimum, it can't slow down in time and **overshoots**,
oscillating back and forth around the bottom before settling.

### Step 2 — Nesterov Accelerated Gradient (NAG): brake before the turn
**Problem:** Momentum decides its next move using the gradient at its *current*
position — but by the time it arrives there, its accumulated velocity has
already carried it further. It's like braking for a curve only after you're
already in it.

**Fix — "the smart driver":** peek ahead to where momentum is *about* to carry
you (w − β·v), and compute the gradient **there** instead of at the current
point:
```
v := β·v + η·∇L(w − β·v)      ← gradient evaluated at the look-ahead point
w := w − v
```
If that look-ahead point is already past the minimum (uphill again), the
gradient there points backward and pulls velocity down *before* the overshoot
happens — braking at the apex of the turn instead of after it.

**Remaining flaw:** NAG is a strictly better Momentum, but it still applies the
exact same learning rate η to **every parameter**. In models with sparse
features (e.g. text/embeddings) some parameters get updated constantly while
others are touched rarely — one global η can't serve both well.

### Step 3 — AdaGrad: give every parameter its own learning rate
**Problem:** a rarely-seen feature needs a *big* nudge on the few occasions it
does appear; a frequently-seen feature should take small, careful steps once
it's already close to its optimum. A single global η can't do both.

**Fix — "the custom explorer":** track, **per parameter**, the running sum of
squared gradients seen so far, and divide that parameter's learning rate by the
square root of that sum:
```
G := G + (∇L)²                    (element-wise, accumulated per parameter)
w := w − (η / √(G + ε)) · ∇L
```
A parameter updated often accumulates a large G → its effective step shrinks
(it's already well-explored). A rarely-touched parameter keeps a small G → its
effective step stays large (it needs bigger exploratory jumps).

**New flaw it introduces:** G only ever **grows** (sum of squares, never
resets). Eventually G becomes huge, η/√G shrinks toward 0, and the parameter
**freezes** — training stalls before reaching the minimum, sometimes very early.

### Step 4 — RMSProp: give AdaGrad short-term memory
**Problem:** AdaGrad's memory is permanent and only accumulates — it needs a way
to "forget" ancient gradients so the denominator can't grow without bound.

**Fix — "the short-term-memory hiker":** replace the running **sum** with an
**exponentially decaying average** of squared gradients — recent gradients
count, old ones fade out:
```
G := β·G + (1−β)·(∇L)²            w := w − (η/√(G+ε))·∇L
```
This keeps AdaGrad's per-parameter adaptivity but prevents the denominator from
blowing up, so the optimizer can keep moving indefinitely instead of freezing.
(🔍 AdaDelta is a closely related fix from the same era with a similar goal.)

**Remaining flaw:** RMSProp adapts step *size* per parameter, but it never
accumulates *velocity* — it has no memory of *direction*, so it doesn't get
Momentum's acceleration-through-flat-regions benefit.

### Step 5 — Adam: merge Momentum's velocity with RMSProp's per-parameter scaling
**Fix — "the ultimate rover":** track **both** an exponentially decaying average
of the gradient (first moment m — "which way am I heading") and of the squared
gradient (second moment v — "how bumpy is this terrain"), then combine them:
```
m := β₁·m + (1−β₁)·∇L          (β₁ = 0.9)   ← momentum-style velocity
v := β₂·v + (1−β₂)·(∇L)²       (β₂ = 0.999) ← RMSProp-style adaptive scale
m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)     ← bias correction (m, v start at 0,
                                      so early estimates are biased toward 0)
w := w − η·m̂/(√v̂ + ε)
```
It glides over bumps using accumulated speed (like the bowling ball) while
independently tuning the step size of every single parameter (like the custom
explorer) — hence "the gold standard" / default optimizer for deep learning.

Interview one-liner: *"Adam = momentum (first moment) + RMSProp (second moment)
+ bias correction; robust default with η = 0.001."*

**Remaining flaw:** Adam's interaction with **L2 weight decay** is subtly wrong
(the decay term gets scaled by the adaptive v, which isn't the intended
behavior) — 🔍 **AdamW** decouples weight decay from the adaptive update and is
the modern default in most deep-learning frameworks.

### The Ultimate Summary Table
| Optimizer | Core idea | Analogy | Flaw that motivated the next step |
|---|---|---|---|
| 1. Vanilla GD | Step directly against the current gradient | The Blind Hiker | Slow on gentle slopes; dead stop at flat/saddle zones |
| 2. Momentum | Accumulate a decaying average of past gradients as velocity | The Bowling Ball | No brakes → overshoots and oscillates at the bottom |
| 3. NAG 🔍 | Compute the gradient at the look-ahead position, not the current one | The Smart Driver | Still one global η for every parameter |
| 4. AdaGrad | Per-parameter η, scaled by the running sum of squared gradients | The Custom Explorer | Sum only grows → η → 0 → training freezes |
| 5. RMSProp | Same per-parameter idea, but a decaying average instead of a sum | The Short-Term-Memory Hiker | No velocity/direction memory — no Momentum-style acceleration |
| 6. **Adam** | Momentum's velocity + RMSProp's per-parameter scaling + bias correction | The Ultimate Rover | Subtly wrong weight-decay interaction → fixed by 🔍 AdamW |

---

## 7. Problems GD Runs Into (and fixes)

| Problem | What it is | Fix |
|---|---|---|
| **Local minima** | Non-convex loss traps | SGD noise, momentum, restarts; in high-D deep nets most bad critical points are saddles, not minima 🔍 |
| **Saddle points** | ∇ = 0 but not a minimum (Hessian mixed eigenvalues) | Momentum/noise pushes through flat directions |
| **Plateaus** | Long flat regions with tiny gradients | Momentum, adaptive optimizers |
| **Vanishing gradients** | Deep nets: repeated chain rule through sigmoids/tanh (max slope 0.25) shrinks gradients toward 0 → early layers stop learning | ReLU, residual connections, batch norm, careful init |
| **Exploding gradients** | Product of large derivatives blows up (common in RNNs) | Gradient clipping, careful init |
| **Non-differentiable points** | e.g. \|x\| at 0, ReLU at 0, L1 penalty | **Subgradient** — any slope between the left and right derivatives is valid (why L1/Lasso can still be optimized) |

### 🔍 Newton's Method — why not use second-order information? (full explanation)

**The idea in 1-D first.** Gradient descent knows only the slope: it steps
downhill by a fixed fraction η of the slope, blind to how sharply the curve
bends. Newton's method uses **both slope and curvature**:

```
GD (1-D):      x := x − η · f'(x)          (η chosen by you, by trial)
Newton (1-D):  x := x − f'(x) / f''(x)     (no η — curvature sets the step!)
```

Dividing by the curvature **auto-sizes the step**:
- Sharp curvature (large f'') → the minimum is close → take a *small* step
- Gentle curvature (small f'') → the valley is wide/flat → take a *big* step

**Magic case:** if f is an exact quadratic, e.g. f(x) = (x−5)², then
f' = 2(x−5), f'' = 2, and from any start x:
`x := x − 2(x−5)/2 = 5` — Newton lands on the minimum in **one step**.
GD from x = 0 with η = 0.1 would need dozens of steps for the same journey.
Near a minimum every smooth function looks approximately quadratic, which is why
Newton converges in so few iterations.

**Multivariable version.** The "slope" becomes the gradient ∇L, and "dividing by
curvature" becomes multiplying by the **inverse Hessian** (curvature in every
direction, including the twists):

```
w := w − H⁻¹ · ∇L(w)
```

Geometrically: H⁻¹ **reshapes the step** — it stretches the move along
flat/shallow directions and shrinks it along steep ones, turning elongated
elliptical contours into effective circles. It fixes GD's zig-zag problem
*mathematically* rather than by tuning η.

**So why doesn't deep learning use it? Count the cost for d parameters:**

| Object | Size / cost | d = 10⁶ (a small neural net) |
|---|---|---|
| Gradient ∇L | d numbers | 10⁶ values — fine |
| Hessian H | **d × d** numbers → O(d²) memory | 10¹² values ≈ **4 TB** just to store |
| Inverting H | **O(d³)** time | 10¹⁸ operations — hopeless |

GD costs O(d) per step; Newton costs O(d²)–O(d³) per step. Newton needs *far
fewer* iterations, but each iteration is astronomically more expensive once d is
large — so first-order methods (SGD/Adam) win for deep learning.

**The practical middle ground — Quasi-Newton / L-BFGS:** never form H at all.
Instead, watch how the *gradient changes between recent iterations* — that change
reveals curvature — and build a cheap low-memory approximation of H⁻¹ from just
the last ~10 (gradient, step) pairs. Cost stays near O(d) per step while gaining
much of Newton's fast convergence.
- Excellent for **small/medium convex problems** (logistic regression on tabular
  data) — this is why **sklearn's LogisticRegression uses `solver='lbfgs'` by
  default**
- Not used for deep nets: the curvature approximation copes poorly with
  mini-batch noise and non-convexity

**Interview one-liner:** *"Newton divides the gradient by curvature (H⁻¹) so
steps are auto-sized and convergence takes few iterations — but storing and
inverting a d×d Hessian is O(d²)/O(d³), impossible for large models; L-BFGS
approximates the curvature from recent gradients and is sklearn's default for
logistic regression."*

---

## 8. Connecting to ML Models (where interviewers take this)

- **Linear regression:** MSE convex → GD finds the global optimum
- **Logistic regression:** no closed form; log-loss convex; gradient works out to
  Σ(σ(wᵀxi) − yi)·xi — same elegant form as linear regression's
- **Neural networks:** backprop = chain rule; loss non-convex; mini-batch + Adam
- **Gradient boosting** (link to your GBDT notes!): gradient descent in
  **function space** — each new tree is a step along the negative gradient of the
  loss w.r.t. predictions, with the learning rate as shrinkage
- **Regularization:** L2 adds 2λw to the gradient ("weight decay" shrinks w each
  step); L1 needs subgradients (kink at 0) and drives weights exactly to zero

---

## 9. Frequently Asked Interview Questions

1. What is a derivative, intuitively and formally? What does it mean for training
   a model?
2. State the chain rule and explain its role in backpropagation.
3. Derive the derivative of the sigmoid; why is σ(1−σ) significant for vanishing
   gradients?
4. What is a partial derivative? Compute ∂f/∂x for f = 3x²y + y³.
5. What is the gradient, and why does −∇f give the steepest descent direction?
6. Write the gradient descent update rule. What happens with a too-large / too-
   small learning rate, and what schedules help?
7. Derive the GD updates for linear regression (MSE) and run one iteration by
   hand on toy data.
8. Batch vs SGD vs mini-batch: cost per update, noise, and why mini-batch is the
   practical default. What is an epoch?
9. How can SGD's noise be *helpful*? (Escaping saddles/shallow minima.)
10. Why does feature scaling speed up gradient descent? (Contour/zig-zag picture.)
11. Tell the optimizer evolution story: momentum → NAG → AdaGrad → RMSProp → Adam.
    For each step, state the specific flaw in the previous method that it fixes.
    What exactly does Adam combine, and why bias correction?
11b. What specific problem does NAG solve that plain Momentum doesn't? Where is
    the gradient evaluated in NAG vs Momentum? 🔍 What flaw does AdamW fix in Adam?
12. What is a convex function? Why do we love convex losses? Name convex and
    non-convex losses in ML. Local Minimum = Global Minimum, No Trapping in Saddle Points,convex loss functions represent the "easy mode" of
    optimization 
14. Local minima vs saddle points vs plateaus — definitions and remedies.
15. Explain vanishing and exploding gradients: cause and fixes.
16. Why is linear regression's closed form sometimes worse than GD? (O(d³).)
17. ReLU and \|x\| aren't differentiable at 0 — how does optimization proceed?
    (Subgradients; why L1 still works and yields sparsity.)
18. 🔍 (Senior) Why not use Newton's method for deep nets? What does L-BFGS do?
19. How is gradient boosting "gradient descent in function space"? (Bridge to
    your Gradient_Boosting notes.)
