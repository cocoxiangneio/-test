# -*- coding: utf-8 -*-
"""GA optimizer for factor weight optimization."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GAOptimizer:
    def __init__(
        self,
        n_generations: int = 50,
        pop_size: int = 50,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
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

        population = self._init_population(n_weights, bounds)
        fitness = np.array([objective_func(ind) for ind in population])

        self.best_fitness_ = float(np.max(fitness))
        self.best_weights_ = population[np.argmax(fitness)].copy()
        self.history_.append(self.best_fitness_)

        for gen in range(self.n_generations):
            parents_idx = self._select(population, fitness)
            offspring = []
            for i in range(0, self.pop_size - 1, 2):
                p1 = population[parents_idx[i]]
                p2 = population[parents_idx[i + 1]]
                if np.random.rand() < self.crossover_prob:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = self._mutate(c1, bounds)
                c2 = self._mutate(c2, bounds)
                offspring.extend([c1, c2])
            while len(offspring) < self.pop_size:
                offspring.append(self._mutate(population[parents_idx[0]].copy(), bounds))

            population = np.array(offspring[:self.pop_size])
            fitness = np.array([objective_func(ind) for ind in population])

            gen_best = float(np.max(fitness))
            if gen_best > self.best_fitness_:
                self.best_fitness_ = gen_best
                self.best_weights_ = population[np.argmax(fitness)].copy()
            self.history_.append(self.best_fitness_)

            logger.debug(f"GA Gen {gen + 1}/{self.n_generations}: best={self.best_fitness_:.4f}")

        return self.best_weights_, self.best_fitness_, self.history_

    def _init_population(self, n: int, bounds: List[Tuple[float, float]]) -> np.ndarray:
        pop = np.zeros((self.pop_size, n))
        for i in range(n):
            low, high = bounds[i]
            pop[:, i] = np.random.uniform(low, high, self.pop_size)
        for i in range(self.pop_size):
            pop[i] /= pop[i].sum()
        return pop

    def _select(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        indices = np.argsort(fitness)[::-1]
        return indices

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha = np.random.rand()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        c1 = np.maximum(c1, 0)
        c2 = np.maximum(c2, 0)
        return c1 / (c1.sum() + 1e-10), c2 / (c2.sum() + 1e-10)

    def _mutate(self, ind: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        for i in range(len(ind)):
            if np.random.rand() < self.mutation_prob:
                ind[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        ind = np.maximum(ind, 0)
        if ind.sum() > 0:
            ind /= ind.sum()
        else:
            ind = np.ones(len(ind)) / len(ind)
        return ind

    def get_convergence(self) -> List[float]:
        return self.history_
