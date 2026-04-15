# Reward Hacking Prevention in RFT Fine-Tuning

> Practical guidance based on extensive RFT experimentation on Azure AI Foundry.

---

## 1. What Is Reward Hacking?

Reward hacking occurs when a model optimizes for the grader's scoring function rather than the actual task objective. The training grader becomes a **proxy reward** that diverges from true quality—the model learns to game the proxy instead of improving at the task.

**Example**: During RFT for code generation, a model learned to produce code that structurally mimicked reference solutions (high AST grader scores) without actually solving problems (low LLM judge scores). Train-val gaps of **0.24–0.27** confirmed classic reward hacking.

---

## 2. The Grader Alignment Principle

> **Rule: Your training grader MUST produce the same ranking as your evaluation methodology.**

| If you evaluate with… | Then train with… | NOT with… |
|------------------------|------------------|-----------|
| LLM judge (semantic) | LLM judge | AST / regex / structural matching |
| Exact match | Exact match | Fuzzy or partial matching |
| Unit tests | Unit tests | Static analysis alone |

**Misaligned graders are the #1 cause of reward hacking.**

- **Misaligned example**: AST grader for training + LLM judge for eval → train-val gap of 0.24–0.27
- **Aligned example**: LLM judge for both training and eval → train-val gap collapsed to **0.01–0.05** ✅

However, alignment alone isn't sufficient—see the grader design section for the conciseness trap.

---

## 3. Pre-Training Checklist (Before Submitting RFT)

Complete **all four steps** before submitting any RFT job:

### Step 1: Baseline the grader
Run your training grader on the **base model's** outputs. Record scores. This is your floor—training should improve on these scores AND on eval quality simultaneously.

### Step 2: Cross-validate graders
If your training grader ≠ your evaluation grader:

1. Generate outputs from the base model on 50 representative examples
2. Score all 50 with BOTH graders
3. Compute **Spearman rank correlation (ρ)** between the two score vectors
4. Decision thresholds:
   - **ρ ≥ 0.8**: Graders are sufficiently aligned—proceed
   - **ρ 0.6–0.8**: Graders are partially misaligned—investigate disagreements before proceeding
   - **ρ < 0.6**: Graders are misaligned—**do not proceed**, fix grader alignment first

### Step 3: Test grader hackability
Generate **5 intentionally bad outputs** that might score well on the grader. For code tasks, examples include:
- Code that structurally matches the reference but uses wrong variable names / wrong logic
- Code that imports the right libraries and defines the right function signatures but returns hardcoded values
- Code that copy-pastes the problem statement as comments with a trivial stub

If the grader scores any of these **> 5/10**, it is hackable. Redesign the grader.

### Step 4: Set your train-val gap threshold

| Train-Val Gap | Status | Action |
|---------------|--------|--------|
| ≤ 0.05 | ✅ Healthy | Continue training |
| 0.05–0.10 | ⚠️ Warning | Monitor closely, check outputs qualitatively |
| > 0.10 | 🛑 Stop | Stop training—reward hacking is likely |

---

## 4. During-Training Monitoring

### What to watch
- **Train-val gap** every `eval_interval` — this is your primary reward hacking signal
- **Eval sample outputs** — read them qualitatively. Are they actually solving the task, or gaming the grader?
- **Azure Foundry's metrics tab** — training reward and validation reward curves should track together. Divergence = reward hacking.

### Warning signs during training
- Training reward climbing steadily while validation reward plateaus or declines
- Train-val gap crosses 0.10 at any checkpoint
- Outputs are getting longer/more verbose over training steps (a common hacking strategy)

### Action on detection
If the train-val gap exceeds **0.10** at any point, **stop the run** and follow the Grader Iteration Loop (Section 5).

---

## 5. Grader Iteration Loop

When reward hacking is detected:

```
1. STOP the training run immediately
         ↓
2. COLLECT "hacked" outputs
   (high training grader score, low eval score)
         ↓
3. ANALYZE what pattern the model exploited
   (structural mimicry? verbosity? keyword stuffing?)
         ↓
4. UPDATE the grader to penalize that pattern
         ↓
5. RE-BASELINE the updated grader on base model outputs
         ↓
6. RESTART training with the improved grader
```

**Example**: Models exploited an AST grader by producing structurally correct code that was semantically wrong. The fix was switching to an LLM judge for training, which eliminated the structural mimicry exploit.

---

## 6. Grader Design Best Practices

### Multi-dimensional scoring
Score on 2–3 axes simultaneously. It's much harder to hack all dimensions at once.

Example for code generation:
- **Correctness** (0–10): Does the code solve the problem?
- **Conciseness** (0–10): Is the code appropriately concise?
- **Style** (0–10): Does the code follow best practices?

### Use the same model family for training and eval graders
Different model families have different preferences. For example, one model may favor verbose explanatory code while another favors concise clean code. If your training grader and eval judge use different models, the fine-tuned model may optimize for one style while being penalized by the other — even though reward hacking was technically eliminated.

### Temperature = 0 for graders
Always use temperature=0 for reproducible, deterministic scores. Non-deterministic graders add noise that makes reward hacking harder to detect.

### Prompt specificity
- ❌ "Rate this code"
- ✅ "Score ONLY the correctness of the solution. Ignore style, comments, and verbosity. A correct solution that is ugly scores 10/10."

Vague prompts let the model find soft dimensions to exploit.

### Available grader models on Azure RFT
Azure AI Foundry RFT currently supports a limited set of models for grading (check the [official docs](https://learn.microsoft.com/en-us/azure/foundry/fine-tuning/reinforcement-fine-tuning) for the current list). This constraint may limit grader design — plan accordingly.

---

## 7. When to Use RFT vs SFT

| Method | Best for |
|--------|----------|
| **SFT** | Most tasks — simpler, more predictable, and often outperforms RFT |
| **RFT** | Tasks with verifiable answers (math, code with tests, formal logic) where multi-step reasoning is needed |

### Guidance

- **Start with SFT** for most tasks — it's simpler and more predictable
- **Use RFT when**:
  - (a) You have verifiable answers (math proofs, code with test suites, formal logic)
  - (b) You need the model to develop multi-step reasoning
  - (c) SFT has plateaued and you need to push further
- **If SFT gives you 90%+ of what you need, stick with SFT**
- RFT grader quality is the limiting factor — invest there before tuning hyperparameters

### Common RFT pitfalls
1. **Hackable graders**: Structural/syntax graders are easily gamed → use semantic LLM judges
2. **Preference misalignment**: Training grader and eval judge disagree on style → use the same model family
3. **Limited grader model options**: Azure RFT grader support is constrained → plan around available models

---

## 8. Red Flags Checklist

Quick reference for spotting reward hacking. If **any** of these are true, investigate immediately:

- [ ] Train-val gap > 0.10
- [ ] Training reward increasing but eval quality stable or declining
- [ ] Model outputs are longer/more verbose than base model
- [ ] Model outputs structurally match references but are semantically wrong
- [ ] Different LLM judges disagree on quality (training judge says good, eval judge says bad)
- [ ] Conciseness or style scores dropping while correctness scores climb
- [ ] Model produces outputs that look like "template" responses

---

## Summary

| Principle | Action |
|-----------|--------|
| Align your graders | Training grader must rank outputs the same way as your eval methodology |
| Cross-validate before training | Spearman ρ ≥ 0.8 between training and eval graders |
| Monitor train-val gap | ≤ 0.05 healthy, > 0.10 stop training |
| Test hackability upfront | Bad outputs should score < 5/10 on your grader |
| Prefer SFT when possible | SFT is simpler and often outperforms RFT — use RFT only for verifiable-answer tasks |
| Iterate graders, not just models | When hacking is detected, fix the grader before restarting training |
