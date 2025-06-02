# DiversityEvaluator

`DiversityEvaluator` is a Python library to evaluate the diversity of sampled solutions for a QUBO (Quadratic Unconstrained Binary Optimization) problem. It uses only the quadratic term \( J \) of the QUBO and computes a diversity score based on graph edit distance (GED) between a graph of ideal (low-energy) solutions and a graph of sampled solutions.

## Features

- Accepts a QUBO defined only by its quadratic coefficients \( J_{ij} \).
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