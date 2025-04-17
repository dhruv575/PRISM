# Portfolio Refinement Through Iterative Sequential Modeling (PRISM)

**Week 11 Update – Hyper‑Parameter Meta‑Optimization Layer**

---

## 1. Problem Statement
Our objective remains to optimize a daily equity portfolio for superior **risk‑adjusted returns** while respecting constraints on drawdown, transaction costs, leverage, and sparsity. Week 11 adds an **outer gradient‑descent loop that _learns_ the optimizer’s own hyper‑parameters** (learning rate, decay, momentum, look‑back window). 

### Why This Matters
* Manual hyper‑parameter tuning is labor‑intensive and brittle; automating it lets the optimiser adapt to changing regimes.
* Separately differentiating **drawdown** and **Sortino** gradients gives finer risk control than a single penalty term.
* An expanded 110‑ticker universe delivers broader sector coverage without abandoning computational tractability.

### Success Metric
**Fitness Score** — unchanged composition, but now evaluated with _meta‑tuned_ hyper‑parameters:
$$\text{Score}=\alpha_1\,\text{Sortino}-\alpha_2\,\text{Max DD}-\alpha_3\,\text{Turnover}-\alpha_4\,\text{L1 Sparsity}$$

### Constraints
* **Leverage** ≤ 1 (current runs are long‑only; leverage extension slated for Week 12).
* **Max Drawdown** ≤ 20 %.
* **Transaction Costs** ≤ 0.1 % per step.
* **Sparsity** – target active positions ≈ 20–25.

### Data Requirements
* Daily adjusted closes from Yahoo Finance.
* Training: 1 Jan 2022 → 31 Dec 2023.
* Testing: 1 Jan 2024 → 17 Apr 2025.

### Potential Pitfalls
* Finite‑difference gradients for hyper‑params introduce noise.
* Larger universe raises missing‑data risk.

---

## 2. Technical Approach

### Mathematical Formulation
The inner optimiser maximises:
$$
\max_{\mathbf w}\;\Big[\alpha_1\,\text{Sortino}(\mathbf w)-\alpha_2\,\text{MaxDD}(\mathbf w)-\alpha_3\,\|\mathbf w-\mathbf w^{\text{prev}}\|_1-\alpha_4\,\|\mathbf w\|_1\Big]
$$
with simplex projection (long‑only). Week 11 keeps $\alpha=(1,1,1,0.5,0.2)$ and introduces a **meta‑objective**:
$$
\max_{\theta\in\{\eta_0,\beta,\mu,\mathcal W\}}\;\text{CumReturn}_{\text{train}}\big(\mathrm{OGD}(\theta)\big)
$$
where $\theta$ are hyper‑parameters and OGD denotes the inner optimiser.

### Algorithm & PyTorch Strategy
* **Inner Loop** – Online gradient descent with momentum and decaying step size.
* **Outer Loop** – Five iterations of finite‑difference gradient ascent over $\theta$ (ε = 10⁻³).
* **Diagnostics** – Per‑step logging of gradient norms, HHI, turnover; outer loop stores `history` for plotting.

### Validation Methods
* **In‑Sample** – Optimise hyper‑params on 2022‑2023.
* **Out‑of‑Sample** – Freeze best $\theta$ and evaluate 2024‑2025.
* Baselines: equal‑weight and Week 9 fixed‑param OGD.

### Resource Requirements
Identical Python stack; runtime ≈ 12 s per outer‑loop iteration on CPU for 110 assets.

---

## 3. Initial Results

### Evidence of Working Implementation
* **Meta‑opt layer** closes in on $\eta_0≈0.05$, $\beta≈0.82$, $\mu≈0.70$, window≈45.
* New plotting utility visualises score and hyper‑param paths.

### Performance Metrics
| Metric | Week 9 Fixed‑Params | Week 11 Meta‑Tuned |
|--------|-------------------:|-------------------:|
| Train Cum. Return | baseline | **+7 pp** |
| Test Cum. Return | baseline | **+4 pp** |
| Max Drawdown (Test) | 24 % | **22 %** |
| Avg. Turnover | 0.28 | **0.22** |
| Active Positions > 0.5 % | 24 | **20** |
| HHI | 1 480 | **1 360** |

### Test Case Results
Meta‑tuned portfolio maintains heavy NVDA/MSFT overweight but trims tail allocations, cutting turnover by ≈ 21 %.

### Current Limitations
* Only 5 meta iterations – likely under‑explores space.
* Drawdown gradient still heuristic; future work adds CVaR.

### Resource Usage Measurements
* Total run time (train + test) ≈ 1 m 10 s on 8‑core CPU.

### Unexpected Challenges
* Hyper‑param gradients occasionally explode when window < 25; clamped to ≥ 30.

---

## 4. Intermediate Results

### Test Case Results
Out‑of‑sample cumulative‑return curve lies above Week 9 throughout 2024–2025 except brief February‑2025 drawdown.

### Portfolio Concentration Analysis
* **Active positions** 20–22.
* **Top‑10 weight** ≈ 75 %.
* **HHI** 1 360 indicates moderate concentration relative to Week 9.

### Unexpected Challenges
* Expanded universe magnifies missing‑data edge cases; dual forward/back fills mitigate.

---

## 5. Next Steps
1. **Risk Layer** – Add CVaR and Omega gradients; back‑test on 2008/2020.
2. **Meta Loop** – Increase iterations; experiment with Adam/Adagrad in hyper‑param space.
3. **Leverage & Shorting** – Activate gross leverage ≤ 1.5, per‑asset short cap ≤ 30 %.
4. **Cross‑Asset Diversification** – Add ETF and Treasury buckets.
5. **Codebase** – Separate core vs. meta modules; YAML config; unit tests.

---

## Literature Review
Refer to Week 9 bibliography for foundational multi‑objective optimisation works; CVaR and meta‑learning papers shortlisted for incorporation next sprint.

