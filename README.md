# DiversityEvaluator

`DiversityEvaluator` is a Python library to evaluate the diversity of sampled solutions for a QUBO (Quadratic Unconstrained Binary Optimization) problem. It uses the QUBO and computes a diversity score based on graph edit distance (GED) between a graph of ideal (low-energy) solutions and a graph of sampled solutions.

The code is still strictly for the ideal solution of QUBO and strictly for the GED, but eventually I would like to make an approximate calculation that is easy for everyone to compute.

## Features

- Accepts a QUBO defined by \( Q_{ij} \).
- Uses a single threshold parameter \( \epsilon \) to define ideal solutions:  
  \[
    \text{Threshold} = E_0 + \epsilon \times |E_0|,
  \]
  where \( E_0 \) is the lowest possible energy over all \( 2^n \) binary states.
- Builds a graph of ideal solutions, connecting states that differ by a Hamming distance of 1.
- Allows registering different sets of sampled solutions (e.g., SA, SQA, or any other sampler).
- Computes the graph edit distance (GED) between the ideal graph and the sampled-solution graph.
- Returns a normalized diversity score:
  \[
    \text{Diversity} = 1 - \frac{\text{GED}}{\bigl(\lvert V_\text{ideal}\rvert + \lvert E_\text{ideal}\rvert + \lvert V_\text{sample}\rvert + \lvert E_\text{sample}\rvert\bigr)}.
  \]

## Requirements

- Python 3.12 or higher (`openjij` does not support 3.13 or higher)
- `networkx`
- `numpy`
- (`openjij` if you want to replicate the SA/SQA sampling example)

## Installation
```zsh
pip install git+https://github.com/hiro712/diversity-evaluator
```

Using `uv`
```zsh
uv add git+https://github.com/hiro712/diversity-evaluator
```

## Quick Start
```python
import numpy as np
import openjij as oj
from diversity_evaluator import DiversityEvaluator

# 1) Random QUBO with n = 12
n = 12
np.random.seed(42)
Q = np.random.randn(n, n)
Q = (Q + Q.T) / 2

# 2) Initialize the evaluator class
evaluator = DiversityEvaluator(Q)

# 3) Sampling using SA / SQA
sa_sample = oj.SASampler().sample_qubo(Q, num_reads=500)
sqa_sample = oj.SQASampler().sample_qubo(Q, num_reads=500)

# 4) Extract bitstrings as a set of tuples from sampling results
def extract_states(sample_set) -> set[tuple[int, ...]]:
    states = set()
    for rec in sample_set.record:
        spin = rec.sample
        state = ((spin + 1) // 2).astype(int)
        tup = tuple(state.tolist())
        states.add(tup)
    return states

sa_states = extract_states(sa_sample)
sqa_states = extract_states(sqa_sample)

# 5) Evaluate SA diversity while changing ε
evaluator.register_sample(sa_states)
for eps in [0.01, 0.02, 0.05, 0.1]:
    evaluator.register_epsilon(eps)
    div = evaluator.evaluate()
    print(f"SA Diversity (ε={eps}): {div:.4f}")

# 6) Evaluate SQA diversity while changing ε
evaluator.register_sample(sqa_states)
for eps in [0.01, 0.02, 0.05, 0.1]:
    evaluator.register_epsilon(eps)
    div = evaluator.evaluate()
    print(f"SQA Diversity (ε={eps}): {div:.4f}")
```
