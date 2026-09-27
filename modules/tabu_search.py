"""
Tabu Search for scheduling optimization
"""
from typing import List, Tuple
import numpy as np
from modules.scheduling_core import Job, Schedule, SchedulingUtils
from modules.dispatching_rules import DispatchingRules


class TabuSearch:
    """Tabu Search algorithm"""
    
    def __init__(self, jobs: List[Job], schedule: Schedule, objective: str, params: dict):
        self.jobs = jobs
        self.schedule = schedule
        self.objective = objective
        self.problem_type = schedule.problem_type
        
        self.tabu_list_size = params.get('tabu_list_size', 10)
        self.max_iterations = params.get('max_iterations', 200)
        
        self.tabu_list = []
        self.current_solution = None
        self.current_fitness = float('inf')
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence = []
        
        # Check if any job has predecessors
        self.has_precedence = any(len(job.predecessors) > 0 for job in jobs)
    
    def solve(self) -> dict:
        """Run tabu search"""
        n = len(self.jobs)
        
        # Initialize with valid order (topological if precedence exists)
        if self.has_precedence:
            self.current_solution = SchedulingUtils.topological_sort(self.jobs)
        else:
            self.current_solution = np.random.permutation(n).tolist()
        
        self.current_fitness = self._evaluate(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness
        
        for iteration in range(self.max_iterations):
            neighbors = self._generate_all_neighbors(self.current_solution)
            
            best_neighbor = None
            best_neighbor_fitness = float('inf')
            best_move = None
            
            for neighbor, move in neighbors:
                # Repair for precedence if needed
                if self.has_precedence:
                    neighbor = SchedulingUtils.repair_precedence(neighbor, self.jobs)
                
                fitness = self._evaluate(neighbor)
                
                is_tabu = self._is_tabu_move(move)
                aspiration_criterion = fitness < self.best_fitness
                
                if (not is_tabu or aspiration_criterion) and fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = fitness
                    best_move = move
            
            if best_neighbor is not None:
                self.current_solution = best_neighbor
                self.current_fitness = best_neighbor_fitness
                
                self._add_to_tabu_list(best_move)
                
                if self.current_fitness < self.best_fitness:
                    self.best_solution = self.current_solution.copy()
                    self.best_fitness = self.current_fitness
            
            self.convergence.append(self.best_fitness)
        
        final_schedule = self.schedule.clone()
        final_schedule.reset()
        ordered_jobs = [final_schedule.jobs[i] for i in self.best_solution]
        DispatchingRules._build_schedule(ordered_jobs, final_schedule)
        
        return {
            'schedule': final_schedule,
            'fitness': self.best_fitness,
            'convergence': self.convergence
        }
    
    def _evaluate(self, solution: List[int]) -> float:
        """Evaluate solution fitness"""
        schedule_clone = self.schedule.clone()
        schedule_clone.reset()
        
        ordered_jobs = [schedule_clone.jobs[i] for i in solution]
        DispatchingRules._build_schedule(ordered_jobs, schedule_clone)
        
        return schedule_clone.get_objective_value(self.objective)
    
    def _generate_all_neighbors(self, solution: List[int]) -> List[Tuple[List[int], Tuple[int, int]]]:
        """Generate all swap neighbors"""
        neighbors = []
        n = len(solution)
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append((neighbor, (i, j)))
        
        return neighbors

    def _generate_neighbors(self, solution: List[int]) -> List[Tuple[List[int], Tuple[int, int]]]:
        """Compatibility entry point for the repository's original Tabu Search API.

        The organized implementation uses the complete swap neighborhood by
        default, which is equivalent to the original solver's default mode.
        """
        return self._generate_all_neighbors(solution)
    
    def _is_tabu_move(self, move: Tuple[int, int]) -> bool:
        """Check if move is in tabu list"""
        return any(
            (m[0] == move[0] and m[1] == move[1]) or (m[0] == move[1] and m[1] == move[0])
            for m in self.tabu_list
        )
    
    def _add_to_tabu_list(self, move: Tuple[int, int]):
        """Add move to tabu list"""
        self.tabu_list.append(move)
        if len(self.tabu_list) > self.tabu_list_size:
            self.tabu_list.pop(0)
