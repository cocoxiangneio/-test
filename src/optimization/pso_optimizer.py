# -*- coding: utf-8 -*-
"""PSO optimizer for factor weight optimization."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PSOOptimizer:
    def __init__(
        self,
        n_particles: int = 50,
        n_iterations: int = 50,
        w_inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: Optional[int] = None,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w_inertia
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.best_weights_: Optional[np.ndarray] = None
        self.best_fitness_: float = -np.inf
        self.history_: List[float] = []

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        n_weights: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, float, List[float]]:
        if self.seed is not None:
            np.random.seed(self.seed)

        bounds = bounds or [(0.0, 1.0) for _ in range(n_weights)]

        positions = np.zeros((self.n_particles, n_weights))
        velocities = np.zeros((self.n_particles, n_weights))
        for i in range(n_weights):
            lo, hi = bounds[i]
            positions[:, i] = np.random.uniform(lo, hi, self.n_particles)

        for i in range(self.n_particles):
            positions[i] /= positions[i].sum()

        fitness = np.array([objective_func(p) for p in positions])
        personal_best_pos = positions.copy()
        personal_best_fit = fitness.copy()
        global_best_idx = int(np.argmax(fitness))
        global_best_pos = positions[global_best_idx].copy()
        global_best_fit = float(fitness[global_best_idx])

        self.best_weights_ = global_best_pos.copy()
        self.best_fitness_ = global_best_fit
        self.history_.append(self.best_fitness_)

        for it in range(self.n_iterations):
            r1, r2 = np.random.rand(self.n_particles, n_weights), np.random.rand(self.n_particles, n_weights)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_pos - positions) +
                          self.c2 * r2 * (global_best_pos - positions))

            for i in range(n_weights):
                lo, hi = bounds[i]
                positions[:, i] += velocities[:, i]
                positions[:, i] = np.clip(positions[:, i], lo, hi)

            for i in range(self.n_particles):
                psum = positions[i].sum()
                if psum > 0:
                    positions[i] /= psum
                else:
                    positions[i] = np.ones(n_weights) / n_weights

            fitness = np.array([objective_func(p) for p in positions])
            improved = fitness > personal_best_fit
            personal_best_fit[improved] = fitness[improved]
            personal_best_pos[improved] = positions[improved]

            if float(np.max(fitness)) > global_best_fit:
                gi = int(np.argmax(fitness))
                global_best_fit = float(fitness[gi])
                global_best_pos = positions[gi].copy()

            self.best_fitness_ = global_best_fit
            self.best_weights_ = global_best_pos.copy()
            self.history_.append(self.best_fitness_)

            logger.debug(f"PSO Iter {it + 1}/{self.n_iterations}: best={self.best_fitness_:.4f}")

        return self.best_weights_, self.best_fitness_, self.history_

    def get_convergence(self) -> List[float]:
        return self.history_
