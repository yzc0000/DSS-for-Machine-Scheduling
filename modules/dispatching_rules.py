"""
Dispatching rules for scheduling problems
"""
from typing import List
from modules.scheduling_core import Job, Machine, Schedule


class DispatchingRules:
    """Collection of dispatching rule algorithms"""
    
    @staticmethod
    def SPT(jobs: List[Job], schedule: Schedule) -> Schedule:
        """Shortest Processing Time"""
        schedule.reset()
        sorted_jobs = sorted(jobs, key=lambda j: j.get_total_processing_time())
        return DispatchingRules._build_schedule(sorted_jobs, schedule)
    
    @staticmethod
    def LPT(jobs: List[Job], schedule: Schedule) -> Schedule:
        """Longest Processing Time"""
        schedule.reset()
        sorted_jobs = sorted(jobs, key=lambda j: j.get_total_processing_time(), reverse=True)
        return DispatchingRules._build_schedule(sorted_jobs, schedule)
    
    @staticmethod
    def EDD(jobs: List[Job], schedule: Schedule) -> Schedule:
        """Earliest Due Date"""
        schedule.reset()
        sorted_jobs = sorted(jobs, key=lambda j: j.due_date if j.due_date is not None else float('inf'))
        return DispatchingRules._build_schedule(sorted_jobs, schedule)
    
    @staticmethod
    def WSPT(jobs: List[Job], schedule: Schedule) -> Schedule:
        """Weighted Shortest Processing Time"""
        schedule.reset()
        sorted_jobs = sorted(jobs, key=lambda j: j.weight / j.get_total_processing_time(), reverse=True)
        return DispatchingRules._build_schedule(sorted_jobs, schedule)
    
    @staticmethod
    def FIFO(jobs: List[Job], schedule: Schedule) -> Schedule:
        """First In First Out (by release date)"""
        schedule.reset()
        sorted_jobs = sorted(jobs, key=lambda j: (j.release_date, j.id))
        return DispatchingRules._build_schedule(sorted_jobs, schedule)

    @staticmethod
    def ERD(jobs: List[Job], schedule: Schedule) -> Schedule:
        """Earliest Release Date."""
        schedule.reset()
        sorted_jobs = sorted(jobs, key=lambda j: (j.release_date, j.id))
        return DispatchingRules._build_schedule(sorted_jobs, schedule)

    @staticmethod
    def WrapAround(jobs: List[Job], schedule: Schedule) -> Schedule:
        """Round-robin assignment for parallel machines."""
        schedule.reset()
        sorted_jobs = sorted(jobs, key=lambda j: j.id)
        return DispatchingRules._build_wrap_around(sorted_jobs, schedule)

    @staticmethod
    def Johnson(jobs: List[Job], schedule: Schedule) -> Schedule:
        """Johnson's optimal sequencing rule for a two-machine flow shop."""
        schedule.reset()
        if schedule.num_machines != 2 or schedule.problem_type != 'F':
            sorted_jobs = sorted(jobs, key=lambda j: j.get_total_processing_time())
            return DispatchingRules._build_schedule(sorted_jobs, schedule)

        first = []
        second = []
        for job in jobs:
            if job.get_processing_time(0) < job.get_processing_time(1):
                first.append(job)
            else:
                second.append(job)
        first.sort(key=lambda j: j.get_processing_time(0))
        second.sort(key=lambda j: j.get_processing_time(1), reverse=True)
        return DispatchingRules._build_flow_shop(first + second, schedule)
    
    @staticmethod
    def _build_schedule(sorted_jobs: List[Job], schedule: Schedule) -> Schedule:
        """Build schedule based on job ordering and problem type.
        When precedence constraints exist, reorder using topological sort while
        respecting the dispatching rule priority as a tiebreaker."""
        
        # Check if any job has precedence constraints
        has_precedence = any(len(job.predecessors) > 0 for job in sorted_jobs)
        
        if has_precedence:
            # Reorder using topological sort with dispatching priority as tiebreaker
            sorted_jobs = DispatchingRules._topological_reorder(sorted_jobs)
        
        schedule.set_job_sequence([j.id for j in sorted_jobs])
        
        if schedule.problem_type == '1':
            return DispatchingRules._build_single_machine(sorted_jobs, schedule)
        elif schedule.problem_type == 'P':
            return DispatchingRules._build_parallel_machines(sorted_jobs, schedule)
        elif schedule.problem_type == 'F':
            return DispatchingRules._build_flow_shop(sorted_jobs, schedule)
        elif schedule.problem_type == 'J':
            return DispatchingRules._build_job_shop(sorted_jobs, schedule)
        elif schedule.problem_type == 'O':
            return DispatchingRules._build_open_shop(sorted_jobs, schedule)
        else:
            return DispatchingRules._build_single_machine(sorted_jobs, schedule)
    
    @staticmethod
    def _topological_reorder(jobs: List[Job]) -> List[Job]:
        """Reorder jobs to respect precedence while keeping dispatching priority as tiebreaker.
        Jobs earlier in the input list have higher priority."""
        job_map = {job.id: job for job in jobs}
        priority = {job.id: i for i, job in enumerate(jobs)}  # Lower = higher priority
        
        # Build in-degree count
        in_degree = {job.id: 0 for job in jobs}
        for job in jobs:
            for pred_id in job.predecessors:
                if pred_id in job_map:
                    in_degree[job.id] += 1
        
        # Build successor list
        successors = {job.id: [] for job in jobs}
        for job in jobs:
            for pred_id in job.predecessors:
                if pred_id in successors:
                    successors[pred_id].append(job.id)
        
        # Kahn's algorithm with priority queue
        ready = [job.id for job in jobs if in_degree[job.id] == 0]
        ready.sort(key=lambda jid: priority[jid])  # Sort by dispatching priority
        
        result = []
        while ready:
            current = ready.pop(0)
            result.append(job_map[current])
            
            # Add successors that are now ready
            new_ready = []
            for succ in successors[current]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    new_ready.append(succ)
            
            # Sort new ready jobs by priority and merge into ready list
            new_ready.sort(key=lambda jid: priority[jid])
            # Insert maintaining priority order
            for jid in new_ready:
                inserted = False
                for i, rid in enumerate(ready):
                    if priority[jid] < priority[rid]:
                        ready.insert(i, jid)
                        inserted = True
                        break
                if not inserted:
                    ready.append(jid)
        
        return result
    
    @staticmethod
    def _build_single_machine(sorted_jobs: List[Job], schedule: Schedule) -> Schedule:
        """Single machine scheduling with precedence constraints"""
        machine = schedule.machines[0]
        current_time = 0.0
        
        # Build job lookup for predecessor completion times
        job_map = {job.id: job for job in sorted_jobs}
        
        for job in sorted_jobs:
            # Start time must respect: machine available, release date, and predecessor completion
            earliest_start = max(current_time, job.release_date)
            
            # Check predecessor completion times
            for pred_id in job.predecessors:
                if pred_id in job_map:
                    pred_job = job_map[pred_id]
                    earliest_start = max(earliest_start, pred_job.completion_time)
            
            job.start_time = earliest_start
            current_time = machine.add_job(job, earliest_start)
            job.completion_time = current_time
        
        schedule.calculate_metrics()
        return schedule
    
    @staticmethod
    def _build_parallel_machines(sorted_jobs: List[Job], schedule: Schedule) -> Schedule:
        """Parallel machines scheduling with precedence constraints"""
        # Build job lookup for predecessor completion times
        job_map = {job.id: job for job in sorted_jobs}
        
        for job in sorted_jobs:
            # Find machine with earliest available time
            earliest_machine = min(schedule.machines, key=lambda m: m.available_time)
            
            # Start time must respect machine, release date, and predecessors
            start_time = max(earliest_machine.available_time, job.release_date)
            
            # Check predecessor completion times
            for pred_id in job.predecessors:
                if pred_id in job_map:
                    pred_job = job_map[pred_id]
                    start_time = max(start_time, pred_job.completion_time)
            
            job.start_time = start_time
            end_time = earliest_machine.add_job(job, start_time)
            job.completion_time = end_time
        
        schedule.calculate_metrics()
        return schedule

    @staticmethod
    def _build_wrap_around(sorted_jobs: List[Job], schedule: Schedule) -> Schedule:
        """Assign jobs to parallel machines in round-robin order."""
        num_machines = schedule.num_machines
        job_map = {job.id: job for job in sorted_jobs}

        for index, job in enumerate(sorted_jobs):
            machine = schedule.machines[index % num_machines]
            start_time = max(machine.available_time, job.release_date)
            for predecessor_id in job.predecessors:
                if predecessor_id in job_map:
                    start_time = max(start_time, job_map[predecessor_id].completion_time)
            job.start_time = start_time
            job.completion_time = machine.add_job(job, start_time)

        schedule.calculate_metrics()
        return schedule
    
    @staticmethod
    def _build_flow_shop(sorted_jobs: List[Job], schedule: Schedule) -> Schedule:
        """Flow shop scheduling (permutation flow shop)"""
        num_machines = schedule.num_machines
        
        for job in sorted_jobs:
            start_time = 0.0
            
            for m in range(num_machines):
                machine = schedule.machines[m]
                # Job can start when both machine is free and previous operation is done
                start_time = max(machine.available_time, start_time)
                
                if m == 0:
                    start_time = max(start_time, job.release_date)
                
                if m == 0:
                    job.start_time = start_time
                
                processing_time = job.get_processing_time(m)
                end_time = start_time + processing_time
                
                machine.add_job(job, start_time)
                start_time = end_time
            
            job.completion_time = start_time
        
        schedule.calculate_metrics()
        return schedule
    
    @staticmethod
    def _build_job_shop(sorted_jobs: List[Job], schedule: Schedule) -> Schedule:
        """Job shop scheduling with job-specific routing (Giffler-Thompson variant)"""
        num_machines = schedule.num_machines
        
        # Track operation index for each job
        job_operations = {job.id: 0 for job in sorted_jobs}
        job_last_completion = {job.id: job.release_date for job in sorted_jobs}
        
        # Calculate total operations (sum of routing lengths, or num_machines if no routing)
        total_operations = 0
        for job in sorted_jobs:
            if job.routing:
                total_operations += len(job.routing)
            else:
                total_operations += num_machines
        
        completed_operations = 0
        
        while completed_operations < total_operations:
            earliest_job = None
            earliest_machine = None
            earliest_time = float('inf')
            earliest_processing = 0
            
            # Find next operation to schedule
            for job in sorted_jobs:
                op_index = job_operations[job.id]
                
                # Determine number of operations for this job
                job_num_ops = len(job.routing) if job.routing else num_machines
                
                if op_index < job_num_ops:
                    # Get machine for this operation from routing (or default to op_index)
                    if job.routing:
                        machine_id = job.routing[op_index]
                    else:
                        machine_id = op_index
                    
                    machine = schedule.machines[machine_id]
                    start_time = max(machine.available_time, job_last_completion[job.id])
                    
                    if start_time < earliest_time:
                        earliest_time = start_time
                        earliest_job = job
                        earliest_machine = machine
                        # Get processing time for this specific machine
                        earliest_processing = job.get_processing_time(machine_id)
            
            if earliest_job:
                op_index = job_operations[earliest_job.id]
                
                if op_index == 0:
                    earliest_job.start_time = earliest_time
                
                # Manually calculate end time and update machine
                end_time = earliest_time + earliest_processing
                from modules.scheduling_core import Task
                earliest_machine.schedule.append(Task(earliest_job, earliest_time, end_time))
                earliest_machine.available_time = end_time
                
                job_operations[earliest_job.id] = op_index + 1
                job_last_completion[earliest_job.id] = end_time
                
                # Check if job is complete
                job_num_ops = len(earliest_job.routing) if earliest_job.routing else num_machines
                if op_index + 1 == job_num_ops:
                    earliest_job.completion_time = end_time
                
                completed_operations += 1
            else:
                break
        
        schedule.calculate_metrics()
        return schedule
    
    @staticmethod
    def _build_open_shop(sorted_jobs: List[Job], schedule: Schedule) -> Schedule:
        """Open shop scheduling (greedy)"""
        operations = []
        
        for job in sorted_jobs:
            for m in range(schedule.num_machines):
                operations.append({
                    'job': job,
                    'machine': m,
                    'processing_time': job.get_processing_time(m)
                })
        
        # Sort operations by processing time
        operations.sort(key=lambda op: op['processing_time'])
        
        job_last_time = {job.id: job.release_date for job in sorted_jobs}
        
        for op in operations:
            job = op['job']
            machine = schedule.machines[op['machine']]
            
            start_time = max(machine.available_time, job_last_time[job.id])
            
            if job.start_time == 0:
                job.start_time = start_time
            
            end_time = machine.add_job(job, start_time)
            job_last_time[job.id] = end_time
            job.completion_time = max(job.completion_time, end_time)
        
        schedule.calculate_metrics()
        return schedule
