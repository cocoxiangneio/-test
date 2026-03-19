# -*- coding: utf-8 -*-
"""NSGA-II multi-objective genetic algorithm optimizer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Individual:
    genes: np.ndarray
    objectives: List[float]
    rank: int
    crowding_distance: float

    def dominates(self, other: "Individual") -> bool:
        at_least_as_good = all(a >= b for a, b in zip(self.objectives, other.objectives))
        strictly_better = any(a > b for a, b in zip(self.objectives, other.objectives))
        return at_least_as_good and strictly_better


class NSGAII:
    def __init__(
        self,
        n_objectives: int = 2,
        pop_size: int = 100,
        n_generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        eta: float = 20.0,
        seed: Optional[int] = None,
    ):
        self.n_objectives = n_objectives
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta = eta
        self.seed = seed
        self.pareto_front_: Optional[List[Individual]] = None

    def _initialize_population(self, bounds: List[Tuple[float, float]]) -> List[Individual]:
        if self.seed is not None:
            np.random.seed(self.seed)
        pop = []
        for _ in range(self.pop_size):
            genes = np.array([random.uniform(lo, hi) for lo, hi in bounds])
            ind = Individual(genes=genes, objectives=[0.0] * self.n_objectives, rank=0, crowding_distance=0.0)
            pop.append(ind)
        return pop

    def _evaluate(self, pop: List[Individual], objectives: List[Callable]) -> None:
        for ind in pop:
            ind.objectives = [obj(ind.genes) for obj in objectives]

    def _fast_non_dominated_sort(self, pop: List[Individual]) -> List[List[Individual]]:
        fronts: List[List[int]] = [[]]
        domination_count: Dict[int, int] = {}
        dominated_set: Dict[int, List[int]] = {}

        for i, p in enumerate(pop):
            domination_count[i] = 0
            dominated_set[i] = []
            for j, q in enumerate(pop):
                if i == j:
                    continue
                if p.dominates(q):
                    dominated_set[i].append(j)
                elif q.dominates(p):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                pop[i].rank = 0
                fronts[0].append(i)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: List[int] = []
            for pi in fronts[i]:
                for qj in dominated_set[pi]:
                    domination_count[qj] -= 1
                    if domination_count[qj] == 0:
                        pop[qj].rank = i + 1
                        next_front.append(qj)
            fronts.append(next_front)
            i += 1
        return [[pop[idx] for idx in front] for front in fronts if front]

    def _crowding_distance(self, front: List[Individual]) -> None:
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return
        for ind in front:
            ind.crowding_distance = 0.0
        for obj_idx in range(self.n_objectives):
            sorted_front = sorted(front, key=lambda x: x.objectives[obj_idx])
            sorted_front[0].crowding_distance = float("inf")
            sorted_front[-1].crowding_distance = float("inf")
            obj_range = sorted_front[-1].objectives[obj_idx] - sorted_front[0].objectives[obj_idx]
            if obj_range == 0:
                obj_range = 1e-10
            for i in range(1, len(sorted_front) - 1):
                sorted_front[i].crowding_distance += (
                    sorted_front[i + 1].objectives[obj_idx] - sorted_front[i - 1].objectives[obj_idx]
                ) / obj_range

    def _tournament_select(self, pop: List[Individual]) -> Individual:
        i, j = random.sample(range(len(pop)), 2)
        a, b = pop[i], pop[j]
        if a.rank < b.rank:
            return a
        if b.rank < a.rank:
            return b
        if a.crowding_distance > b.crowding_distance:
            return a
        return b

    def _crossover(self, p1: Individual, p2: Individual, bounds: List[Tuple[float, float]]) -> Tuple[Individual, Individual]:
        if random.random() < self.crossover_prob:
            u = np.random.rand(len(p1.genes))
            beta = np.where(u <= 0.5, (2 * u) ** (1 / (self.eta + 1)), (2 * (1 - u)) ** (-1 / (self.eta + 1)))
            c1_genes = 0.5 * ((1 + beta) * p1.genes + (1 - beta) * p2.genes)
            c2_genes = 0.5 * ((1 - beta) * p1.genes + (1 + beta) * p2.genes)
            c1_genes = np.clip(c1_genes, [b[0] for b in bounds], [b[1] for b in bounds])
            c2_genes = np.clip(c2_genes, [b[0] for b in bounds], [b[1] for b in bounds])
        else:
            c1_genes = p1.genes.copy()
            c2_genes = p2.genes.copy()
        return (
            Individual(c1_genes, [0.0] * self.n_objectives, 0, 0.0),
            Individual(c2_genes, [0.0] * self.n_objectives, 0, 0.0),
        )

    def _mutate(self, ind: Individual, bounds: List[Tuple[float, float]]) -> Individual:
        mutated_genes = ind.genes.copy()
        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_prob:
                delta_l = (mutated_genes[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
                delta_r = (bounds[i][1] - mutated_genes[i]) / (bounds[i][1] - bounds[i][0])
                shift = random.random()
                if shift < 0.5:
                    delta_q = (2 * shift + (1 - 2 * shift) * (1 - delta_l) ** (self.eta + 1)) ** (1 / (self.eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - shift) + 2 * (shift - 0.5) * (1 - delta_r) ** (self.eta + 1)) ** (1 / (self.eta + 1))
                mutated_genes[i] += delta_q * (bounds[i][1] - bounds[i][0])
                mutated_genes[i] = np.clip(mutated_genes[i], bounds[i][0], bounds[i][1])
        return Individual(mutated_genes, [0.0] * self.n_objectives, 0, 0.0)

    def _create_offspring(self, pop: List[Individual], bounds: List[Tuple[float, float]]) -> List[Individual]:
        offspring = []
        while len(offspring) < self.pop_size:
            p1 = self._tournament_select(pop)
            p2 = self._tournament_select(pop)
            c1, c2 = self._crossover(p1, p2, bounds)
            offspring.append(self._mutate(c1, bounds))
            if len(offspring) < self.pop_size:
                offspring.append(self._mutate(c2, bounds))
        return offspring[: self.pop_size]

    def optimize(
        self,
        objectives: List[Callable],
        bounds: List[Tuple[float, float]],
    ) -> List[Tuple[np.ndarray, List[float]]]:
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        pop = self._initialize_population(bounds)
        self._evaluate(pop, objectives)
        fronts = self._fast_non_dominated_sort(pop)
        for front in fronts:
            self._crowding_distance(front)

        for gen in range(self.n_generations):
            offspring = self._create_offspring(pop, bounds)
            self._evaluate(offspring, objectives)
            combined = pop + offspring
            fronts = self._fast_non_dominated_sort(combined)
            new_pop: List[Individual] = []
            for front in fronts:
                self._crowding_distance(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    remaining = self.pop_size - len(new_pop)
                    sorted_front = sorted(front, key=lambda x: -x.crowding_distance)
                    new_pop.extend(sorted_front[:remaining])
                    break
            pop = new_pop

        fronts = self._fast_non_dominated_sort(pop)
        self.pareto_front_ = fronts[0] if fronts else []
        return [(ind.genes, ind.objectives) for ind in self.pareto_front_]
