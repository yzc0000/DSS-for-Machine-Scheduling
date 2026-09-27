"""
Core scheduling data structures and utilities
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np


@dataclass
class Job:
    """Represents a job with processing times and constraints"""
    id: int
    processing_times: Union[float, List[float]]  # Single value or list for multi-machine
    due_date: Optional[float] = None
    weight: float = 1.0
    release_date: float = 0.0
    completion_time: float = 0.0
    start_time: float = 0.0
    predecessors: List[int] = field(default_factory=list)  # List of predecessor job IDs
    routing: List[int] = field(default_factory=list)  # Machine visit order for job shop (e.g., [2, 0, 1] = M3→M1→M2)
    
    def get_total_processing_time(self) -> float:
        """Get sum of all processing times"""
        if isinstance(self.processing_times, list):
            return sum(self.processing_times)
        return self.processing_times
    
    def get_processing_time(self, machine_id: int = 0) -> float:
        """Get processing time for specific machine"""
        if isinstance(self.processing_times, list):
            return self.processing_times[machine_id] if machine_id < len(self.processing_times) else 0
        return self.processing_times
    
    def get_routing_processing_time(self, operation_index: int) -> float:
        """Get processing time for operation at given index in routing"""
        if not self.routing or operation_index >= len(self.routing):
            return self.get_processing_time(operation_index)
        machine_id = self.routing[operation_index]
        return self.get_processing_time(machine_id)


    def clone(self):
        """Create a deep copy of the job"""
        new_job = Job(
            self.id,
            self.processing_times.copy() if isinstance(self.processing_times, list) else self.processing_times,
            self.due_date,
            self.weight,
            self.release_date
        )
        new_job.predecessors = self.predecessors.copy()
        new_job.routing = self.routing.copy()
        return new_job


@dataclass
class Task:
    """Represents a scheduled task on a machine"""
    job: Job
    start_time: float
    end_time: float


class Machine:
    """Represents a machine with its schedule"""
    
    def __init__(self, id: int):
        self.id = id
        self.schedule: List[Task] = []
        self.available_time = 0.0
    
    def add_job(self, job: Job, start_time: float) -> float:
        """Add job to machine schedule and return completion time"""
        processing_time = job.get_processing_time(self.id)
        end_time = start_time + processing_time
        
        self.schedule.append(Task(job, start_time, end_time))
        self.available_time = end_time
        return end_time
    
    def reset(self):
        """Reset machine schedule"""
        self.schedule = []
        self.available_time = 0.0


class Schedule:
    """Represents a complete schedule with metrics"""
    
    def __init__(self, jobs: List[Job], problem_type: str, num_machines: int = 1):
        self.jobs = jobs
        self.problem_type = problem_type
        self.num_machines = num_machines
        self.machines = [Machine(i) for i in range(num_machines)]
        self.job_sequence = []
        
        # Metrics
        self.makespan = 0.0
        self.total_completion_time = 0.0
        self.total_weighted_completion_time = 0.0
        self.max_lateness = float('-inf')
        self.total_tardiness = 0.0
    
    def set_job_sequence(self, sequence: List[int]):
        """Set the job processing sequence"""
        self.job_sequence = sequence
    
    def calculate_metrics(self):
        """Calculate all scheduling metrics"""
        self.makespan = max(machine.available_time for machine in self.machines)
        
        self.total_completion_time = sum(job.completion_time for job in self.jobs)
        self.total_weighted_completion_time = sum(
            job.weight * job.completion_time for job in self.jobs
        )
        
        # Calculate lateness and tardiness
        self.max_lateness = float('-inf')
        self.total_tardiness = 0.0
        
        for job in self.jobs:
            if job.due_date is not None:
                lateness = job.completion_time - job.due_date
                self.max_lateness = max(self.max_lateness, lateness)
                self.total_tardiness += max(0, lateness)
        
        return {
            'makespan': self.makespan,
            'total_completion_time': self.total_completion_time,
            'total_weighted_completion_time': self.total_weighted_completion_time,
            'max_lateness': self.max_lateness if self.max_lateness != float('-inf') else None,
            'total_tardiness': self.total_tardiness
        }
    
    def get_objective_value(self, objective: str) -> float:
        """Get value for specific objective function"""
        if objective == 'Cmax':
            return self.makespan
        elif objective == 'sumCi':
            return self.total_completion_time
        elif objective == 'sumwiCi':
            return self.total_weighted_completion_time
        elif objective == 'Lmax':
            return self.max_lateness
        elif objective == 'sumTi':
            return self.total_tardiness
        return self.makespan
    
    def reset(self):
        """Reset all machines and job completion times"""
        for machine in self.machines:
            machine.reset()
        for job in self.jobs:
            job.completion_time = 0.0
            job.start_time = 0.0
    
    def clone(self):
        """Create a copy of this schedule"""
        # Deep copy jobs to ensure isolation
        new_jobs = [job.clone() for job in self.jobs]
        new_schedule = Schedule(new_jobs, self.problem_type, self.num_machines)
        new_schedule.job_sequence = self.job_sequence.copy()
        return new_schedule


class SchedulingUtils:
    """Utility functions for scheduling algorithms"""
    
    @staticmethod
    def random_permutation(n: int) -> List[int]:
        """Generate random permutation of integers 0 to n-1"""
        return np.random.permutation(n).tolist()
    
    @staticmethod
    def topological_sort(jobs: List[Job]) -> List[int]:
        """Topological sort of jobs based on precedence constraints.
        Returns a valid ordering where predecessors come before successors."""
        n = len(jobs)
        job_map = {job.id: job for job in jobs}
        
        # Build in-degree count and adjacency list
        in_degree = {job.id: len(job.predecessors) for job in jobs}
        successors = {job.id: [] for job in jobs}
        
        for job in jobs:
            for pred_id in job.predecessors:
                if pred_id in successors:
                    successors[pred_id].append(job.id)
        
        # Kahn's algorithm
        queue = [job.id for job in jobs if in_degree[job.id] == 0]
        result = []
        
        while queue:
            # Sort by some priority (e.g., processing time for SPT-like ordering)
            queue.sort(key=lambda jid: job_map[jid].get_total_processing_time())
            current = queue.pop(0)
            result.append(current)
            
            for succ in successors[current]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        
        if len(result) != n:
            # Cycle detected, return original order
            return [job.id for job in jobs]
        
        return result
    
    @staticmethod
    def is_valid_precedence_order(sequence: List[int], jobs: List[Job]) -> bool:
        """Check if a sequence respects all precedence constraints."""
        job_map = {job.id: job for job in jobs}
        position = {job_id: i for i, job_id in enumerate(sequence)}
        
        for job_id in sequence:
            job = job_map.get(job_id)
            if job:
                for pred_id in job.predecessors:
                    if pred_id in position and position[pred_id] >= position[job_id]:
                        return False  # Predecessor comes after this job
        return True
    
    @staticmethod
    def repair_precedence(sequence: List[int], jobs: List[Job]) -> List[int]:
        """Repair a sequence to respect precedence constraints."""
        if SchedulingUtils.is_valid_precedence_order(sequence, jobs):
            return sequence
        
        # Use topological sort as fallback
        job_list = [job for job in jobs if job.id in sequence]
        return SchedulingUtils.topological_sort(job_list)

    @staticmethod
    def topological_sort_indices(jobs: List[Job]) -> List[int]:
        """Return a precedence-feasible ordering as list indices."""
        id_to_index = {job.id: index for index, job in enumerate(jobs)}
        return [id_to_index[job_id] for job_id in SchedulingUtils.topological_sort(jobs)]

    @staticmethod
    def insert_move(individual: List[int]) -> List[int]:
        """Remove one item and insert it at another position."""
        result = individual.copy()
        if len(result) < 2:
            return result
        remove_position = np.random.randint(0, len(result))
        item = result.pop(remove_position)
        insert_position = np.random.randint(0, len(result) + 1)
        result.insert(insert_position, item)
        return result

    @staticmethod
    def swap_adjacent(individual: List[int]) -> List[int]:
        """Swap one randomly selected adjacent pair."""
        result = individual.copy()
        if len(result) < 2:
            return result
        position = np.random.randint(0, len(result) - 1)
        result[position], result[position + 1] = result[position + 1], result[position]
        return result


    @staticmethod
    def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover for permutation representation"""
        n = len(parent1)
        child = [-1] * n
        
        # Select random substring from parent1
        start = np.random.randint(0, n)
        end = np.random.randint(start, n)
        
        # Copy substring to child
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Fill remaining from parent2
        child_pos = (end + 1) % n
        for i in range(n):
            parent_pos = (end + 1 + i) % n
            gene = parent2[parent_pos]
            
            if gene not in child:
                child[child_pos] = gene
                child_pos = (child_pos + 1) % n
        
        return child
    
    @staticmethod
    def swap_mutation(individual: List[int]) -> List[int]:
        """Swap two random positions in permutation"""
        mutated = individual.copy()
        n = len(mutated)
        i, j = np.random.choice(n, 2, replace=False)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    @staticmethod
    def tournament_selection(population: List[List[int]], 
                            fitness_scores: List[float], 
                            tournament_size: int = 3) -> List[int]:
        """Tournament selection for genetic algorithm"""
        n = len(population)
        competitors = np.random.choice(n, tournament_size, replace=False)
        best = min(competitors, key=lambda i: fitness_scores[i])
        return population[best]


class ScheduleValidator:
    """Validate machine capacity, release, precedence, and flow constraints."""

    @staticmethod
    def validate(schedule: 'Schedule') -> dict:
        errors = []

        for machine in schedule.machines:
            tasks = sorted(machine.schedule, key=lambda task: task.start_time)
            for current, following in zip(tasks, tasks[1:]):
                if current.end_time > following.start_time + 0.001:
                    errors.append(
                        f"Overlap on M{machine.id}: Job {current.job.id} ends at "
                        f"{current.end_time:.2f} but Job {following.job.id} starts at "
                        f"{following.start_time:.2f}"
                    )

        for job in schedule.jobs:
            if job.start_time < job.release_date - 0.001:
                errors.append(
                    f"Job {job.id} starts at {job.start_time:.2f} before release "
                    f"{job.release_date:.2f}"
                )

        jobs_by_id = {job.id: job for job in schedule.jobs}
        for job in schedule.jobs:
            for predecessor_id in job.predecessors:
                predecessor = jobs_by_id.get(predecessor_id)
                if predecessor and job.start_time < predecessor.completion_time - 0.001:
                    errors.append(
                        f"Precedence violated: Job {job.id} starts before "
                        f"predecessor {predecessor_id} completes"
                    )

        if schedule.problem_type == 'F':
            for job in schedule.jobs:
                previous_end = None
                for machine in schedule.machines:
                    task = next((task for task in machine.schedule if task.job.id == job.id), None)
                    if task is not None and previous_end is not None and task.start_time < previous_end - 0.001:
                        errors.append(f"Flow order violated for Job {job.id} on M{machine.id}")
                    if task is not None:
                        previous_end = task.end_time

        return {'valid': not errors, 'errors': errors}

    @staticmethod
    def validate_and_print(schedule: 'Schedule') -> bool:
        result = ScheduleValidator.validate(schedule)
        if result['valid']:
            print('Schedule is valid!')
        else:
            print('Schedule has errors:')
            for error in result['errors']:
                print(f'  - {error}')
        return result['valid']
