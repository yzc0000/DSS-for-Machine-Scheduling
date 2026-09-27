
import random
from modules.scheduling_core import Job, Schedule
from modules.dispatching_rules import DispatchingRules

def test_job_shop_variance():
    # Create a simple 3x3 Job Shop instance
    # Job 0: M1(10) -> M2(5) -> M3(8)
    # Job 1: M2(8) -> M1(10) -> M3(5)
    # Job 2: M3(5) -> M1(8) -> M2(10)
    
    # Note: The current implementation assumes a fixed routing M1->M2->M3... 
    # or does it? Let's check how processing times are stored.
    # Job.processing_times is a list [p_j1, p_j2, ...]
    # _build_job_shop uses job_operations[job.id] to track progress.
    # machine_id = next_op % num_machines
    # So it assumes cyclic routing 0->1->2->0... or just 0->1->2 if ops < machines?
    
    # Let's inspect _build_job_shop logic again.
    # total_operations = len(sorted_jobs) * num_machines
    # next_op < num_machines
    # machine_id = next_op % num_machines
    # Yes, it assumes Flow Shop routing (M1 -> M2 -> M3) but allows jobs to be prioritized differently at each step.
    # Wait, if it's fixed routing, it's a Flow Shop?
    # Job Shop usually implies different routings for different jobs.
    # If the code enforces M1->M2->M3 for ALL jobs, then it's effectively a Flow Shop problem 
    # where we are just deciding the sequence on machines.
    
    # Let's test with 5 jobs, 3 machines.
    jobs = []
    num_jobs = 5
    num_machines = 3
    
    for i in range(num_jobs):
        p_times = [random.randint(5, 20) for _ in range(num_machines)]
        jobs.append(Job(i, p_times))
        
    schedule = Schedule(jobs, 'J', num_machines)
    
    # Test 1: Forward order
    schedule.reset()
    jobs_fwd = sorted(jobs, key=lambda j: j.id)
    DispatchingRules._build_job_shop(jobs_fwd, schedule)
    cmax_fwd = schedule.makespan
    print(f"Forward Order Cmax: {cmax_fwd}")
    
    # Test 2: Reverse order
    schedule.reset()
    jobs_rev = sorted(jobs, key=lambda j: j.id, reverse=True)
    DispatchingRules._build_job_shop(jobs_rev, schedule)
    cmax_rev = schedule.makespan
    print(f"Reverse Order Cmax: {cmax_rev}")
    
    # Test 3: Random orders
    variances = []
    for _ in range(10):
        schedule.reset()
        jobs_rnd = jobs.copy()
        random.shuffle(jobs_rnd)
        DispatchingRules._build_job_shop(jobs_rnd, schedule)
        variances.append(schedule.makespan)
        
    print(f"Random Orders Cmax: {variances}")
    print(f"Unique results: {len(set(variances))}")

if __name__ == "__main__":
    test_job_shop_variance()
