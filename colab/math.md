Below is a **step-by-step mathematical flow** that covers the *entire* pipeline:

```
          ┌──────────────────┐
Prompt p ─┤  0  MEMORY SEARCH ├──────────────────────┐
          └──────────────────┘                      │
                                                   ▼
          ┌──────────────────┐        ┌──────────────────────────┐
prefix    │ 1  MEMORY WEIGHTS│        │2  DUAL-BIAS PREPARATION  │
tokens c  └──────────────────┘        └──────────────────────────┘
                                                   │
                                                   ▼
          ┌──────────────────┐        ┌──────────────────────────┐
          │3  TRANSFORMER    │        │ 4  DRAFT SAMPLING (N)    │
          └──────────────────┘        └──────────────────────────┘
                                                   │
                                                   ▼
          ┌──────────────────┐
          │5  SIEVE (rank)   │
          └──────────────────┘
```

---

## 0 Memory search  (CSCSR retriever)

*For every stored memory* $m^{(j)},j=1..M$:

```
semantic = cos( SBERT(m^j) , SBERT(prompt) )
lexical  = lex_overlap( m^j , prompt )           ∈ [0,1]

R_j      = λ · semantic + (1−λ) · lexical        (blend, λ∈[0,1])
```

### pick top-N

Choose the N memories with the largest **R\_j**.

---

## 1 Memory-weight diffusion and ordering

```
w_j    = max( R_j^τ , σ )          τ>0   (exponential “diffuse down”)
```

Then **sort** the N memories by their original timestamp
`oldest → newest`.
After tokenising you have

```
c^1_1 … c^1_{ℓ1} | c^2_1 … | … | c^N_{ℓN}
weights:  w_1     … w_1     | w_2 …       w_N
```

`L_mem = ℓ1+…+ℓN`.

---

## 2 Dual-bias preparation

### 2.1 Memory-weight *tape*

```
tape[i] = corresponding w_j           for i = 1 … L_mem
tape[i] = 0                           for i > L_mem   (user prompt)
```

### 2.2 Attention-mask multiplier

*(applied once, on the 1st forward pass)*

```
scale_i = 1 + α · tape[i]          if s_mask = 1
scale_i = 1                        if s_mask = 0
```

### 2.3 Logit-bias vector

*(applied every decoding step)*

```
b[v] = β · Σ_j  w_j · 1{ token-id v appears in memory j }   if s_bias = 1
b[v] = 0                                                    if s_bias = 0
```

---

## 3 Transformer pass with soft fusion

*Step 0 (past =None)*

```
emb_i = E[ x_i ] · scale_i         ──► transformer θ ──► hidden
```

*Step t ≥ 0*

```
logits_t = transformer_output_t  +  b
logits_t = logits_t / T                         (temperature)
```

---

## 4 Sampling N drafts

For each draft `d = 1..N_d`:

```
probs_t   = softmax( logits_t )   (plus top-k / nucleus / repetition penalty)
y_t  ~  probs_t                   (sample next id)
```

Collect full draft Y^(d).

---

## 5 Sieve / ranking

For every draft Y:

```
SBERT-sim   s  = cos( SBERT(Y), SBERT(prompt) )
LM-loss     L  = −log P_θ( Y | prompt )
ρ           = λ_s · L  +  (1−λ_s) · (1 − s)      λ_s ∈ [0,1]
```

*Blacklist*: if Y hits forbidden phrase ⇒ ρ = ∞.

**Select** the draft with *minimum* ρ.

---

### Parameter cheat-sheet

| symbol  | role                                   | usual range |
| ------- | -------------------------------------- | ----------- |
| λ       | semantic vs lexical blend in retrieval | 0.4 – 0.7   |
| τ       | diffusion slope (↓ bigger ⇒ flatter)   | 0.8 – 3     |
| σ       | weight floor after diffusion           | 0 – 0.2     |
| α       | attention-mask strength                | 0 – 0.6     |
| β       | logit-bias magnitude                   | 0 – 5       |
| s\_mask | 0/1 switch for attention mask          | {0,1}       |
| s\_bias | 0/1 switch for logit bias              | {0,1}       |
| T       | softmax temperature                    | 0.7 – 1.2   |
| λ\_s    | LM-loss vs SBERT in sieve              | 0.3 – 0.7   |
| N\_d    | number of parallel drafts              | 2 – 8       |

---

### One-line summary

1. **Rank & diffuse** memories → chronological tape
2. **Soft-fuse** them:
   *sequence-level* `scale_i = 1+α·w`  **and**
   *token-level*   `b[v] = β·w`
3. **Sample multiple drafts** with these biases
4. **Pick** the lowest‐cost draft by *LM loss ⊕ SBERT similarity*, after blacklist.
