import itertools
import time

import networkx as nx
import numpy as np


class DiversityEvaluator:
    """
    A class to evaluate solution diversity for a QUBO using only the quadratic term J.
    Uses a single epsilon and a single sample set to compute diversity based on graph edit distance (GED).

    Usage example:

        # (1) Define J for an n-variable QUBO
        n = 12
        J = {(i, j): np.random.uniform(-1, 1) for i in range(n) for j in range(i + 1, n)}

        # (2) Initialize (optionally providing epsilon and an initial sample set)
        evaluator = DiversityEvaluator(J=J, epsilon=0.05, sample_states=sa_states)
        # If you want a one-time diversity calculation immediately:
        diversity_score = evaluator.evaluate()

        # (3) To use a different epsilon for the same sample:
        evaluator.register_epsilon(0.10)
        diversity_score = evaluator.evaluate()

        # (4) To replace the sample set and compute diversity again:
        evaluator.register_sample(sqa_states)
        evaluator.register_epsilon(0.02)
        diversity_score = evaluator.evaluate()

    Args:
        J (dict[tuple[int, int], float]):
            The quadratic term of the QUBO. Keys are (i, j) with i < j, values are coefficients w_{ij}.
        epsilon (float, optional):
            The threshold parameter for ideal solutions, used as E0 + epsilon * |E0|.
            If None, call register_epsilon() before evaluating.
        sample_states (set[tuple[int, ...]], optional):
            The initial set of sampled solutions (each is a length-n tuple of 0/1).
            If None, call register_sample() before evaluating.
    """

    def __init__(
        self,
        J: dict[tuple[int, int], float],
        epsilon: float = None,
        sample_states: set[tuple[int, ...]] = None,
    ):
        self.J = J
        self.epsilon = epsilon
        self.sample_states = sample_states

        # Infer number of variables n from J
        self.n = self._infer_num_variables(J)

        # Precompute all 2^n states and their energies, find E0
        self._prepare_all_states()

        # Prepare the ideal solution graph if epsilon is provided
        self.G_ideal = None
        self._ideal_node_count = 0
        self._ideal_edge_count = 0
        if self.epsilon is not None:
            self._prepare_ideal(self.epsilon)

        # Prepare the sample graph if sample_states is provided
        self.G_sample = None
        self._sample_node_count = 0
        self._sample_edge_count = 0
        if self.sample_states is not None:
            self._prepare_sample(self.sample_states)

    @staticmethod
    def _infer_num_variables(J: dict[tuple[int, int], float]) -> int:
        """
        Infer the number of variables n from the keys of J.
        """
        if not J:
            return 0
        max_index = max(max(i, j) for (i, j) in J.keys())
        return max_index + 1

    def _energy_of(self, state: tuple[int, ...]) -> float:
        """
        Compute the energy of a binary state (tuple of 0/1) under the quadratic term J.

        E = sum_{(i,j) in J} J[(i,j)] * state[i] * state[j]
        """
        e = 0.0
        for (i, j), w in self.J.items():
            e += w * state[i] * state[j]
        return e

    def _prepare_all_states(self):
        """
        Enumerate all 2^n binary states, compute their energies, and find the minimum energy E0.
        """
        self._all_states = list(itertools.product([0, 1], repeat=self.n))
        self._all_energies = np.array([self._energy_of(s) for s in self._all_states])
        self.E0 = float(self._all_energies.min())

    def _build_graph(self, state_list: list[tuple[int, ...]]) -> nx.Graph:
        """
        Build a graph where nodes represent binary states (as strings),
        and edges connect states whose Hamming distance is 1.
        """
        G = nx.Graph()
        labels = [''.join(map(str, s)) for s in state_list]
        G.add_nodes_from(labels)
        for a in state_list:
            sa = ''.join(map(str, a))
            for b in state_list:
                sb = ''.join(map(str, b))
                if sum(ai != bi for ai, bi in zip(a, b)) == 1:
                    G.add_edge(sa, sb)
        return G

    def _prepare_ideal(self, epsilon: float):
        """
        Compute the set of ideal solutions (energy <= E0 + epsilon * |E0|) and build its graph.
        Cache the node count and edge count of the ideal graph.
        """
        threshold = self.E0 + abs(self.E0) * epsilon
        self._ideal_states = [
            s for s, e in zip(self._all_states, self._all_energies) if e <= threshold
        ]
        self.G_ideal = self._build_graph(self._ideal_states)
        self._ideal_node_count = self.G_ideal.number_of_nodes()
        self._ideal_edge_count = self.G_ideal.number_of_edges()

    def _prepare_sample(self, sample_states: set[tuple[int, ...]]):
        """
        Build and cache the graph representation of the sample set.
        """
        states = list(sample_states)
        self.G_sample = self._build_graph(states)
        self._sample_node_count = self.G_sample.number_of_nodes()
        self._sample_edge_count = self.G_sample.number_of_edges()

    def register_epsilon(self, epsilon: float):
        """
        Update epsilon, recompute the ideal solution graph, and cache its properties.
        """
        self.epsilon = epsilon
        self._prepare_ideal(epsilon)

    def register_sample(self, sample_states: set[tuple[int, ...]]):
        """
        Update the sample solution set, rebuild the sample graph, and cache its properties.
        """
        self.sample_states = sample_states
        self._prepare_sample(sample_states)

    def _calc_ged(self, timeout: float = 60.0) -> float:
        """
        Compute the graph edit distance (GED) between the ideal graph and the sample graph.
        If the exact distance is not found within the timeout, return the best intermediate value.
        """
        G1 = self.G_ideal
        G2 = self.G_sample
        ub = (
            self._ideal_node_count
            + self._ideal_edge_count
            + self._sample_node_count
            + self._sample_edge_count
        )
        ged = nx.graph_edit_distance(G1, G2, upper_bound=ub, timeout=timeout)
        if ged is None:
            best = None
            start = time.monotonic()
            for d in nx.optimize_graph_edit_distance(G1, G2, upper_bound=ub):
                best = d
                if time.monotonic() - start > timeout:
                    break
            if best is None:
                best = ub
            ged = best
        if not isinstance(ged, (int, float)):
            ged = min(ged)
        return float(ged)

    def evaluate(self) -> float:
        """
        Compute and return the diversity score:
            diversity = 1 - (GED / normalization_constant),
        where normalization_constant = (ideal_node_count + ideal_edge_count + sample_node_count + sample_edge_count).

        Raises:
            RuntimeError: if epsilon or sample set has not been registered.
        """
        if self.G_ideal is None:
            raise RuntimeError("Epsilon is not set. Call register_epsilon() first.")
        if self.G_sample is None:
            raise RuntimeError("Sample set is not registered. Call register_sample() first.")

        ged = self._calc_ged()
        nc = (
            self._ideal_node_count
            + self._ideal_edge_count
            + self._sample_node_count
            + self._sample_edge_count
        )
        diversity = 1.0 - ged / nc if nc > 0 else 0.0
        return diversity