"""
Genetic Algorithm for Scheduling Problems
"""
from typing import List, Dict
import numpy as np
from modules.scheduling_core import Job, Schedule, SchedulingUtils
from modules.dispatching_rules import DispatchingRules


class GeneticAlgorithm:
    """Genetic Algorithm solver for scheduling optimization"""
    
    def __init__(self, jobs: List[Job], schedule: Schedule, objective: str, params: Dict):
        self.jobs = jobs
        self.schedule = schedule
        self.objective = objective
        self.population_size = params.get('population_size', 50)
        self.generations = params.get('generations', 100)
        self.crossover_rate = params.get('crossover_rate', 0.8)
        self.mutation_rate = params.get('mutation_rate', 0.1)
        self.elite_size = max(1, self.population_size // 10)
        # Check if any job has predecessors
        self.has_precedence = any(len(job.predecessors) > 0 for job in jobs)
    
    def solve(self) -> Dict:
        """Run genetic algorithm and return best solution"""
        n = len(self.jobs)
        convergence = []
        
        # Initialize population - use topological sort if precedence exists
        if self.has_precedence:
            base_order = SchedulingUtils.topological_sort(self.jobs)
            population = [base_order.copy() for _ in range(self.population_size)]
            for i in range(1, self.population_size):
                individual = SchedulingUtils.random_permutation(n)
                population[i] = SchedulingUtils.repair_precedence(individual, self.jobs)
        else:
            population = [SchedulingUtils.random_permutation(n) for _ in range(self.population_size)]
            spt_order = sorted(range(n), key=lambda i: self.jobs[i].get_total_processing_time())
            population[0] = spt_order
        
        # Evaluate initial population
        fitness_scores = [self._evaluate(individual) for individual in population]
        
        best_individual = population[np.argmin(fitness_scores)]
        best_fitness = min(fitness_scores)
        convergence.append(best_fitness)
        
        for generation in range(self.generations):
            new_population = []
            
            # Elitism: Keep best individuals
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = SchedulingUtils.tournament_selection(population, fitness_scores)
                parent2 = SchedulingUtils.tournament_selection(population, fitness_scores)
                
                if np.random.random() < self.crossover_rate:
                    child = SchedulingUtils.order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                if np.random.random() < self.mutation_rate:
                    child = SchedulingUtils.swap_mutation(child)
                
                # Repair for precedence if needed
                if self.has_precedence:
                    child = SchedulingUtils.repair_precedence(child, self.jobs)
                
                new_population.append(child)
            
            population = new_population[:self.population_size]
            fitness_scores = [self._evaluate(individual) for individual in population]
            
            gen_best_idx = np.argmin(fitness_scores)
            if fitness_scores[gen_best_idx] < best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            convergence.append(best_fitness)
        
        best_schedule = self._build_schedule(best_individual)
        
        return {
            'schedule': best_schedule,
            'fitness': best_fitness,
            'convergence': convergence,
            'best_sequence': best_individual
        }
    
    def _evaluate(self, individual: List[int]) -> float:
        """Evaluate fitness of an individual (lower is better)"""
        schedule = self._build_schedule(individual)
        return schedule.get_objective_value(self.objective)
    
    def _build_schedule(self, sequence: List[int]) -> Schedule:
        """Build schedule from job sequence"""
        new_schedule = self.schedule.clone()
        sorted_jobs = [new_schedule.jobs[i] for i in sequence]
        DispatchingRules._build_schedule(sorted_jobs, new_schedule)
        return new_schedule
