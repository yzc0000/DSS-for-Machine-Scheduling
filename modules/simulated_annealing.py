"""
Simulated Annealing for Scheduling Problems
"""
from typing import List, Dict
import numpy as np
import math
from modules.scheduling_core import Job, Schedule, SchedulingUtils
from modules.dispatching_rules import DispatchingRules


class SimulatedAnnealing:
    """Simulated Annealing solver for scheduling optimization"""
    
    def __init__(self, jobs: List[Job], schedule: Schedule, objective: str, params: Dict):
        self.jobs = jobs
        self.schedule = schedule
        self.objective = objective
        self.initial_temp = params.get('initial_temp', 1000)
        self.cooling_rate = params.get('cooling_rate', 0.95)
        self.iterations_per_temp = params.get('iterations_per_temp', 100)
        self.min_temp = params.get('min_temp', 1)
        # Check if any job has predecessors
        self.has_precedence = any(len(job.predecessors) > 0 for job in jobs)
    
    def solve(self) -> Dict:
        """Run simulated annealing and return best solution"""
        n = len(self.jobs)
        convergence = []
        
        # Start with valid order (topological if precedence, SPT otherwise)
        if self.has_precedence:
            current = SchedulingUtils.topological_sort(self.jobs)
        else:
            current = sorted(range(n), key=lambda i: self.jobs[i].get_total_processing_time())
        
        current_fitness = self._evaluate(current)
        
        best = current.copy()
        best_fitness = current_fitness
        
        temperature = self.initial_temp
        iteration = 0
        
        while temperature > self.min_temp:
            for _ in range(self.iterations_per_temp):
                neighbor = self._get_neighbor(current)
                
                # Repair for precedence if needed
                if self.has_precedence:
                    neighbor = SchedulingUtils.repair_precedence(neighbor, self.jobs)
                
                neighbor_fitness = self._evaluate(neighbor)
                delta = neighbor_fitness - current_fitness
                
                if delta < 0:
                    current = neighbor
                    current_fitness = neighbor_fitness
                else:
                    acceptance_prob = math.exp(-delta / temperature)
                    if np.random.random() < acceptance_prob:
                        current = neighbor
                        current_fitness = neighbor_fitness
                
                if current_fitness < best_fitness:
                    best = current.copy()
                    best_fitness = current_fitness
                
                iteration += 1
            
            convergence.append(best_fitness)
            temperature *= self.cooling_rate
        
        best_schedule = self._build_schedule(best)
        
        return {
            'schedule': best_schedule,
            'fitness': best_fitness,
            'convergence': convergence,
            'best_sequence': best,
            'iterations': iteration
        }
    
    def _get_neighbor(self, solution: List[int]) -> List[int]:
        """Generate neighbor by swapping two random positions"""
        neighbor = solution.copy()
        n = len(neighbor)
        i, j = np.random.choice(n, 2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def _evaluate(self, individual: List[int]) -> float:
        """Evaluate fitness of a solution (lower is better)"""
        schedule = self._build_schedule(individual)
        return schedule.get_objective_value(self.objective)
    
    def _build_schedule(self, sequence: List[int]) -> Schedule:
        """Build schedule from job sequence"""
        new_schedule = self.schedule.clone()
        sorted_jobs = [new_schedule.jobs[i] for i in sequence]
        DispatchingRules._build_schedule(sorted_jobs, new_schedule)
        return new_schedule
