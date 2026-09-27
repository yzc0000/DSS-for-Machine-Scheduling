"""
Ant Colony Optimization for Scheduling Problems
"""
from typing import List, Dict
import numpy as np
from modules.scheduling_core import Job, Schedule, SchedulingUtils
from modules.dispatching_rules import DispatchingRules


class AntColonyOptimization:
    """Ant Colony Optimization solver for scheduling optimization"""
    
    def __init__(self, jobs: List[Job], schedule: Schedule, objective: str, params: Dict):
        self.jobs = jobs
        self.schedule = schedule
        self.objective = objective
        self.num_ants = params.get('num_ants', 20)
        self.iterations = params.get('iterations', 100)
        self.alpha = params.get('alpha', 1.0)
        self.beta = params.get('beta', 2.0)
        self.evaporation_rate = params.get('evaporation_rate', 0.5)
        self.q = 100
        
        self.n = len(jobs)
        self.pheromone = np.ones((self.n, self.n)) * 0.1
        
        # Check if any job has predecessors
        self.has_precedence = any(len(job.predecessors) > 0 for job in jobs)
    
    def solve(self) -> Dict:
        """Run ACO and return best solution"""
        convergence = []
        
        best_solution = None
        best_fitness = float('inf')
        
        heuristic = np.zeros(self.n)
        for i, job in enumerate(self.jobs):
            heuristic[i] = 1.0 / max(job.get_total_processing_time(), 0.001)
        
        for iteration in range(self.iterations):
            solutions = []
            fitness_values = []
            
            for ant in range(self.num_ants):
                solution = self._construct_solution(heuristic)
                
                # Repair for precedence if needed
                if self.has_precedence:
                    solution = SchedulingUtils.repair_precedence(solution, self.jobs)
                
                fitness = self._evaluate(solution)
                solutions.append(solution)
                fitness_values.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()
            
            self._update_pheromones(solutions, fitness_values)
            
            convergence.append(best_fitness)
        
        if best_solution is None:
            best_solution = list(range(self.n))
        
        best_schedule = self._build_schedule(best_solution)
        
        return {
            'schedule': best_schedule,
            'fitness': best_fitness,
            'convergence': convergence,
            'best_sequence': best_solution
        }
    
    def _construct_solution(self, heuristic: np.ndarray) -> List[int]:
        """Construct a solution using probabilistic selection"""
        solution = []
        available = set(range(self.n))
        
        current_pos = 0
        
        while available:
            if len(solution) == 0:
                probs = np.zeros(self.n)
                for j in available:
                    probs[j] = (heuristic[j] ** self.beta)
            else:
                probs = np.zeros(self.n)
                last_job = solution[-1]
                for j in available:
                    pheromone_val = self.pheromone[last_job, j] ** self.alpha
                    heuristic_val = heuristic[j] ** self.beta
                    probs[j] = pheromone_val * heuristic_val
            
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                for j in available:
                    probs[j] = 1.0 / len(available)
            
            next_job = np.random.choice(self.n, p=probs)
            solution.append(next_job)
            available.remove(next_job)
        
        return solution
    
    def _update_pheromones(self, solutions: List[List[int]], fitness_values: List[float]):
        """Update pheromone trails"""
        self.pheromone *= (1 - self.evaporation_rate)
        
        for solution, fitness in zip(solutions, fitness_values):
            if fitness > 0:
                deposit = self.q / fitness
            else:
                deposit = self.q
            
            for i in range(len(solution) - 1):
                from_job = solution[i]
                to_job = solution[i + 1]
                self.pheromone[from_job, to_job] += deposit
        
        self.pheromone = np.maximum(self.pheromone, 0.01)
    
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
