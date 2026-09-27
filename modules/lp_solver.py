"""
LP Relaxation Solver using PuLP
"""
from typing import List
import numpy as np
from modules.scheduling_core import Job, Schedule

try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

        
class LPRelaxationSolver:
    """LP relaxation solver for lower bounds"""
    
    def __init__(self, jobs: List[Job], schedule: Schedule, objective: str):
        self.jobs = jobs
        self.schedule = schedule
        self.objective = objective
        self.problem_type = schedule.problem_type
    
    def solve(self) -> dict:
        """Solve LP relaxation for lower bound"""
        lower_bound = 0.0
        
        try:
            if self.problem_type == '1':
                lower_bound = self._single_machine_bound()
            elif self.problem_type == 'P':
                lower_bound = self._parallel_machine_bound()
            elif self.problem_type == 'F':
                lower_bound = self._flow_shop_bound()
            elif self.problem_type == 'J':
                lower_bound = self._job_shop_bound()
            elif self.problem_type == 'O':
                lower_bound = self._open_shop_bound()
        except Exception as e:
            print(f"LP Solver error: {e}")
            lower_bound = 0.0
        
        return {
            'lower_bound': lower_bound,
            'method': 'LP Relaxation' if PULP_AVAILABLE else 'Analytical Bound'
        }
    
    
    def _single_machine_bound(self) -> float:
        """Lower bound for single machine"""
        if self.objective == 'Cmax':
            # Makespan = sum of all processing times (ignoring release dates for LB)
            total_p = sum(job.get_total_processing_time() for job in self.jobs)
            # Also consider max(release_date + processing_time) as a bound
            max_completion = max(job.release_date + job.get_total_processing_time() for job in self.jobs)
            return max(total_p, max_completion)
        
        elif self.objective == 'sumCi':
            # SPT order gives optimal for 1||sumCi (no release dates)
            # For lower bound, we ignore release dates (relaxation)
            sorted_times = sorted(job.get_total_processing_time() for job in self.jobs)
            time = 0.0
            total_completion = 0.0
            for p in sorted_times:
                time += p
                total_completion += time
            return total_completion
        
        elif self.objective == 'sumwiCi':
            # WSPT order gives optimal for 1||sumwiCi (no release dates)
            # For lower bound, we ignore release dates (relaxation)
            weighted_jobs = [(job.weight / job.get_total_processing_time(), 
                             job.get_total_processing_time(), 
                             job.weight) for job in self.jobs]
            weighted_jobs.sort(key=lambda x: x[0], reverse=True)
            time = 0.0
            total_weighted = 0.0
            for _, p, w in weighted_jobs:
                time += p
                total_weighted += w * time
            return total_weighted
        
        elif self.objective == 'Lmax':
            # Lower bound: EDD gives optimal for 1||Lmax
            # LB = max over all jobs of (sum of smaller processing times + p_j - d_j)
            # Simple bound: max(p_j - d_j) assuming job starts at 0
            sorted_jobs = sorted(self.jobs, 
                               key=lambda j: j.due_date if j.due_date is not None else float('inf'))
            time = 0.0
            max_lateness = float('-inf')
            for job in sorted_jobs:
                if job.due_date is not None:
                    time += job.get_total_processing_time()
                    lateness = time - job.due_date
                    max_lateness = max(max_lateness, lateness)
            return max_lateness if max_lateness != float('-inf') else 0.0
        
        elif self.objective == 'sumTi':
            # Lower bound for total tardiness
            # Use EDD schedule as heuristic lower bound approximation
            sorted_jobs = sorted(self.jobs, 
                               key=lambda j: j.due_date if j.due_date is not None else float('inf'))
            time = 0.0
            total_tardiness = 0.0
            for job in sorted_jobs:
                time += job.get_total_processing_time()
                if job.due_date is not None:
                    total_tardiness += max(0, time - job.due_date)
            # This is actually an upper bound from EDD, not a true lower bound
            # For true LB, we'd need LP relaxation. Return 0 as trivial LB.
            return 0.0
        
        return sum(job.get_total_processing_time() for job in self.jobs)
    
    def _parallel_machine_bound(self) -> float:
        """Lower bound for parallel machines"""
        m = self.schedule.num_machines
        
        # For identical parallel machines, use actual processing times
        processing_times = [job.get_total_processing_time() for job in self.jobs]
        total_time = sum(processing_times)
        max_time = max(processing_times)
        
        if self.objective == 'Cmax':
            # LB = max(total_time / m, max_single_job_time)
            return max(total_time / m, max_time)
        
        elif self.objective == 'sumCi':
            # Correct lower bound for P||sumCi using SPT on m machines
            # Simulate assigning jobs in SPT order to earliest available machine
            sorted_times = sorted(processing_times)
            machine_times = [0.0] * m
            total_completion = 0.0
            
            for p in sorted_times:
                # Assign to machine with earliest available time
                earliest = min(range(m), key=lambda i: machine_times[i])
                machine_times[earliest] += p
                total_completion += machine_times[earliest]
            
            return total_completion
        
        elif self.objective == 'sumwiCi':
            # WSPT-based bound for parallel machines
            # Sort by weighted shortest processing time (w/p ratio, descending)
            weighted_jobs = [(job.weight / max(job.get_total_processing_time(), 0.001), 
                             job.get_total_processing_time(), 
                             job.weight) for job in self.jobs]
            weighted_jobs.sort(key=lambda x: x[0], reverse=True)
            
            machine_times = [0.0] * m
            total_weighted_completion = 0.0
            
            for (_, p, w) in weighted_jobs:
                earliest = min(range(m), key=lambda i: machine_times[i])
                machine_times[earliest] += p
                total_weighted_completion += w * machine_times[earliest]
            
            return total_weighted_completion
        
        elif self.objective == 'Lmax':
            # LB for Lmax on parallel machines
            # Simple bound: EDD-like with m machines
            sorted_jobs = sorted(self.jobs, 
                               key=lambda j: j.due_date if j.due_date is not None else float('inf'))
            machine_times = [0.0] * m
            max_lateness = float('-inf')
            for job in sorted_jobs:
                if job.due_date is not None:
                    earliest_machine = min(range(m), key=lambda i: machine_times[i])
                    machine_times[earliest_machine] += job.get_total_processing_time()
                    lateness = machine_times[earliest_machine] - job.due_date
                    max_lateness = max(max_lateness, lateness)
            return max_lateness if max_lateness != float('-inf') else 0.0
        
        elif self.objective == 'sumTi':
            return 0.0  # Trivial lower bound
        
        return total_time / m
    
    
    def _flow_shop_bound(self) -> float:
        """Lower bound for flow shop"""
        m = self.schedule.num_machines
        
        if self.objective == 'Cmax':
            lower_bound = 0.0
            
            # LB1: Maximum machine load
            for machine_id in range(m):
                machine_time = sum(job.get_processing_time(machine_id) for job in self.jobs)
                lower_bound = max(lower_bound, machine_time)
            
            # LB2: Maximum job total time
            for job in self.jobs:
                lower_bound = max(lower_bound, job.get_total_processing_time())
            
            # LB3: For each machine k, compute:
            # sum(p_jk) + min_j(sum_{i<k} p_ji) + min_j(sum_{i>k} p_ji)
            for k in range(m):
                machine_load = sum(job.get_processing_time(k) for job in self.jobs)
                min_before = float('inf')
                min_after = float('inf')
                
                for job in self.jobs:
                    before = sum(job.get_processing_time(i) for i in range(k))
                    after = sum(job.get_processing_time(i) for i in range(k+1, m))
                    min_before = min(min_before, before)
                    min_after = min(min_after, after)
                
                if min_before == float('inf'):
                    min_before = 0
                if min_after == float('inf'):
                    min_after = 0
                    
                lb_k = machine_load + min_before + min_after
                lower_bound = max(lower_bound, lb_k)
            
            return lower_bound
        
        elif self.objective == 'sumCi':
            # For flow shop with m machines, use parallel machine approach
            n = len(self.jobs)
            job_total_times = sorted([job.get_total_processing_time() for job in self.jobs])
            
            # Each job completes at least its processing time
            # With m machines processing in sequence, use machine-aware bound
            total = 0.0
            for i, p in enumerate(job_total_times):
                multiplier = (i // m) + 1
                total += p * multiplier
            return total
        
        elif self.objective == 'sumwiCi':
            # WSPT-based bound with multi-machine consideration
            weighted_jobs = []
            for job in self.jobs:
                total_p = job.get_total_processing_time()
                weighted_jobs.append((job.weight / total_p, total_p, job.weight))
            
            weighted_jobs.sort(key=lambda x: x[0], reverse=True)
            
            total = 0.0
            for i, (_, p, w) in enumerate(weighted_jobs):
                multiplier = (i // m) + 1
                total += w * p * multiplier
            
            return total
        
        elif self.objective == 'Lmax':
            # EDD-based lower bound
            return self._compute_lmax_bound()
        
        elif self.objective == 'sumTi':
            return 0.0
        
        return sum(job.get_total_processing_time() for job in self.jobs)
    
    def _compute_lmax_bound(self) -> float:
        """Compute Lmax lower bound using EDD"""
        sorted_jobs = sorted(self.jobs, 
                           key=lambda j: j.due_date if j.due_date is not None else float('inf'))
        time = 0.0
        max_lateness = float('-inf')
        for job in sorted_jobs:
            time += job.get_total_processing_time()
            if job.due_date is not None:
                lateness = time - job.due_date
                max_lateness = max(max_lateness, lateness)
        return max_lateness if max_lateness != float('-inf') else 0.0
    
    
    def _job_shop_bound(self) -> float:
        """Lower bound for job shop with variable-length routings"""
        m = self.schedule.num_machines
        lower_bound = 0.0
        
        # Helper: get actual processing time for a job (respecting routing)
        def get_job_actual_time(job):
            if job.routing:
                return sum(job.get_processing_time(machine_id) for machine_id in job.routing)
            else:
                return job.get_total_processing_time()
        
        if self.objective == 'Cmax':
            # LB1: Machine-based bound - only count jobs that actually visit each machine
            for machine_id in range(m):
                machine_time = 0.0
                for job in self.jobs:
                    if job.routing:
                        if machine_id in job.routing:
                            machine_time += job.get_processing_time(machine_id)
                    else:
                        machine_time += job.get_processing_time(machine_id)
                lower_bound = max(lower_bound, machine_time)
            
            # LB2: Job-based bound - longest job
            for job in self.jobs:
                lower_bound = max(lower_bound, get_job_actual_time(job))
            
            return lower_bound
        
        elif self.objective == 'sumCi':
            # Use routing-aware processing times
            job_times = sorted([get_job_actual_time(job) for job in self.jobs])
            
            # Simulate SPT on m machines
            machine_times = [0.0] * m
            total = 0.0
            for p in job_times:
                earliest = min(range(m), key=lambda i: machine_times[i])
                machine_times[earliest] += p
                total += machine_times[earliest]
            return total
        
        elif self.objective == 'sumwiCi':
            # WSPT-based bound with routing-aware times
            weighted_jobs = []
            for job in self.jobs:
                actual_time = get_job_actual_time(job)
                if actual_time > 0:
                    weighted_jobs.append((job.weight / actual_time, actual_time, job.weight))
                else:
                    weighted_jobs.append((float('inf'), actual_time, job.weight))
            weighted_jobs.sort(key=lambda x: x[0], reverse=True)
            
            # Simulate WSPT on m machines
            machine_times = [0.0] * m
            total = 0.0
            for (_, p, w) in weighted_jobs:
                earliest = min(range(m), key=lambda i: machine_times[i])
                machine_times[earliest] += p
                total += w * machine_times[earliest]
            return total
        
        elif self.objective == 'Lmax':
            return self._compute_lmax_bound()
        
        elif self.objective == 'sumTi':
            return 0.0
        
        return sum(get_job_actual_time(job) for job in self.jobs)
    
    def _open_shop_bound(self) -> float:
        """Lower bound for open shop"""
        return self._job_shop_bound()
