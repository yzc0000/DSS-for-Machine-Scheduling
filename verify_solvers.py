"""
Comprehensive Verification Tests for Scheduling DSS
====================================================

This file contains test cases with small problem instances where
optimal solutions can be calculated by hand/brute force.

Test Cases:
1. Single Machine with 3 jobs - All permutations (3! = 6)
2. Parallel Machines with 2 machines, 4 jobs
3. Flow Shop with 2 machines, 3 jobs
4. Single Machine with Lmax objective (3 jobs)
"""

import sys
import itertools
from modules.scheduling_core import Job, Schedule
from modules.dispatching_rules import DispatchingRules
from modules.lp_solver import LPRelaxationSolver
from modules.tabu_search import TabuSearch


def brute_force_single_machine(jobs, objective='Cmax'):
    """
    Brute force all permutations for single machine scheduling.
    Returns optimal sequence and objective value.
    """
    n = len(jobs)
    best_sequence = None
    best_value = float('inf')
    all_results = []
    
    for perm in itertools.permutations(range(n)):
        schedule = Schedule([j.clone() for j in jobs], '1', 1)
        schedule.reset()
        
        ordered_jobs = [schedule.jobs[i] for i in perm]
        DispatchingRules._build_schedule(ordered_jobs, schedule)
        
        value = schedule.get_objective_value(objective)
        all_results.append((perm, value))
        
        if value < best_value:
            best_value = value
            best_sequence = perm
    
    return best_sequence, best_value, all_results


def brute_force_flow_shop(jobs, num_machines, objective='Cmax'):
    """
    Brute force all permutations for flow shop scheduling.
    """
    n = len(jobs)
    best_sequence = None
    best_value = float('inf')
    all_results = []
    
    for perm in itertools.permutations(range(n)):
        schedule = Schedule([j.clone() for j in jobs], 'F', num_machines)
        schedule.reset()
        
        ordered_jobs = [schedule.jobs[i] for i in perm]
        DispatchingRules._build_schedule(ordered_jobs, schedule)
        
        value = schedule.get_objective_value(objective)
        all_results.append((perm, value))
        
        if value < best_value:
            best_value = value
            best_sequence = perm
    
    return best_sequence, best_value, all_results


def test_single_machine_cmax():
    """
    Test Case 1: Single Machine | Cmax
    ==================================
    
    Jobs:
    - Job 0: p = 5
    - Job 1: p = 3
    - Job 2: p = 7
    
    For single machine Cmax, the optimal value is always sum(p) = 5+3+7 = 15
    regardless of sequence (no idle time between jobs).
    
    Lower bound should equal 15.
    """
    print("=" * 60)
    print("TEST 1: Single Machine | Cmax")
    print("=" * 60)
    
    jobs = [
        Job(0, 5, None, 1, 0),
        Job(1, 3, None, 1, 0),
        Job(2, 7, None, 1, 0)
    ]
    
    # Expected optimal: sum(p) = 15
    expected_optimal = 15
    
    # Brute force
    best_seq, best_val, all_results = brute_force_single_machine(jobs, 'Cmax')
    print("All permutations (should all give Cmax=15):")
    for seq, val in all_results:
        print(f"  Sequence {seq}: Cmax = {val}")
    
    print(f"\nOptimal: Sequence {best_seq}, Cmax = {best_val}")
    
    # Lower bound
    schedule = Schedule([j.clone() for j in jobs], '1', 1)
    lp_solver = LPRelaxationSolver(jobs, schedule, 'Cmax')
    lb_result = lp_solver.solve()
    print(f"Lower Bound: {lb_result['lower_bound']}")
    
    # Dispatching rule
    schedule2 = Schedule([j.clone() for j in jobs], '1', 1)
    DispatchingRules.SPT(schedule2.jobs, schedule2)
    spt_cmax = schedule2.get_objective_value('Cmax')
    print(f"SPT Cmax: {spt_cmax}")
    
    # Verify
    assert lb_result['lower_bound'] == expected_optimal, f"LB should be {expected_optimal}"
    assert best_val == expected_optimal, f"Optimal should be {expected_optimal}"
    assert spt_cmax == expected_optimal, f"SPT should give {expected_optimal}"
    
    print("\n[PASS] TEST 1 PASSED: LB = Optimal = SPT = 15")
    sys.stdout.flush()
    return True


def test_single_machine_sumci():
    """
    Test Case 2: Single Machine | sumCi
    ====================================
    
    Jobs:
    - Job 0: p = 5
    - Job 1: p = 3
    - Job 2: p = 7
    
    SPT order gives optimal: sort by p ascending = (1, 0, 2)
    
    SPT order: Job1(p=3) -> Job0(p=5) -> Job2(p=7)
    C1 = 3, C0 = 3+5 = 8, C2 = 8+7 = 15
    sumCi = 3 + 8 + 15 = 26
    
    Let's verify other orders are worse:
    - (0,1,2): C0=5, C1=8, C2=15, sum=28
    - (0,2,1): C0=5, C2=12, C1=15, sum=32
    - (1,0,2): C1=3, C0=8, C2=15, sum=26 <- optimal
    - etc.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Single Machine | sumCi")
    print("=" * 60)
    
    jobs = [
        Job(0, 5, None, 1, 0),
        Job(1, 3, None, 1, 0),
        Job(2, 7, None, 1, 0)
    ]
    
    # Brute force
    best_seq, best_val, all_results = brute_force_single_machine(jobs, 'sumCi')
    
    print("All permutations:")
    for seq, val in sorted(all_results, key=lambda x: x[1]):
        marker = " <- optimal" if val == best_val else ""
        print(f"  Sequence {seq}: sumCi = {val}{marker}")
    
    print(f"\nOptimal: Sequence {best_seq}, sumCi = {best_val}")
    
    # Lower bound
    schedule = Schedule([j.clone() for j in jobs], '1', 1)
    lp_solver = LPRelaxationSolver(jobs, schedule, 'sumCi')
    lb_result = lp_solver.solve()
    print(f"Lower Bound: {lb_result['lower_bound']}")
    
    # SPT should give optimal for 1||sumCi
    schedule2 = Schedule([j.clone() for j in jobs], '1', 1)
    DispatchingRules.SPT(schedule2.jobs, schedule2)
    spt_sumci = schedule2.get_objective_value('sumCi')
    print(f"SPT sumCi: {spt_sumci}")
    
    # Verify
    assert lb_result['lower_bound'] == best_val, f"LB should equal optimal for 1||sumCi"
    assert spt_sumci == best_val, f"SPT should give optimal for 1||sumCi"
    
    print("\n[PASS] TEST 2 PASSED: LB = Optimal = SPT = 26")
    sys.stdout.flush()
    return True


def test_single_machine_lmax():
    """
    Test Case 3: Single Machine | Lmax
    ===================================
    
    Jobs:
    - Job 0: p = 5, d = 8
    - Job 1: p = 3, d = 10
    - Job 2: p = 4, d = 6
    
    EDD order gives optimal for 1||Lmax.
    EDD order: sort by due date = (2, 0, 1) with d = (6, 8, 10)
    
    EDD schedule:
    - Job2: C2 = 4, L2 = 4 - 6 = -2
    - Job0: C0 = 4+5 = 9, L0 = 9 - 8 = 1
    - Job1: C1 = 9+3 = 12, L1 = 12 - 10 = 2
    Lmax = max(-2, 1, 2) = 2
    
    Let's verify:
    - (0,1,2): C0=5,C1=8,C2=12 -> L=(-3,-2,6) -> Lmax=6
    - (2,0,1): C2=4,C0=9,C1=12 -> L=(-2,1,2) -> Lmax=2 <- optimal
    """
    print("\n" + "=" * 60)
    print("TEST 3: Single Machine | Lmax")
    print("=" * 60)
    
    jobs = [
        Job(0, 5, 8, 1, 0),   # p=5, d=8
        Job(1, 3, 10, 1, 0),  # p=3, d=10
        Job(2, 4, 6, 1, 0)    # p=4, d=6
    ]
    
    # Brute force
    best_seq, best_val, all_results = brute_force_single_machine(jobs, 'Lmax')
    
    print("All permutations:")
    for seq, val in sorted(all_results, key=lambda x: x[1]):
        marker = " <- optimal" if val == best_val else ""
        print(f"  Sequence {seq}: Lmax = {val}{marker}")
    
    print(f"\nOptimal: Sequence {best_seq}, Lmax = {best_val}")
    
    # Lower bound
    schedule = Schedule([j.clone() for j in jobs], '1', 1)
    lp_solver = LPRelaxationSolver(jobs, schedule, 'Lmax')
    lb_result = lp_solver.solve()
    print(f"Lower Bound: {lb_result['lower_bound']}")
    
    # EDD should give optimal for 1||Lmax
    schedule2 = Schedule([j.clone() for j in jobs], '1', 1)
    DispatchingRules.EDD(schedule2.jobs, schedule2)
    edd_lmax = schedule2.get_objective_value('Lmax')
    print(f"EDD Lmax: {edd_lmax}")
    
    # Verify
    assert lb_result['lower_bound'] <= best_val, "LB should be <= optimal"
    assert edd_lmax == best_val, "EDD should give optimal for 1||Lmax"
    
    print(f"\n[PASS] TEST 3 PASSED: LB ({lb_result['lower_bound']}) <= Optimal = EDD = {best_val}")
    sys.stdout.flush()
    return True


def test_single_machine_wspt():
    """
    Test Case 4: Single Machine | sumwiCi
    ======================================
    
    Jobs:
    - Job 0: p = 6, w = 2 (ratio = 0.33)
    - Job 1: p = 3, w = 3 (ratio = 1.0)  <- highest
    - Job 2: p = 4, w = 1 (ratio = 0.25)
    
    WSPT order: sort by w/p descending = (1, 0, 2) with ratios (1.0, 0.33, 0.25)
    
    WSPT schedule:
    - Job1: C1 = 3, w*C = 3*3 = 9
    - Job0: C0 = 3+6 = 9, w*C = 2*9 = 18
    - Job2: C2 = 9+4 = 13, w*C = 1*13 = 13
    sumwiCi = 9 + 18 + 13 = 40
    """
    print("\n" + "=" * 60)
    print("TEST 4: Single Machine | sumwiCi")
    print("=" * 60)
    
    jobs = [
        Job(0, 6, None, 2, 0),  # p=6, w=2
        Job(1, 3, None, 3, 0),  # p=3, w=3
        Job(2, 4, None, 1, 0)   # p=4, w=1
    ]
    
    # Brute force
    best_seq, best_val, all_results = brute_force_single_machine(jobs, 'sumwiCi')
    
    print("All permutations:")
    for seq, val in sorted(all_results, key=lambda x: x[1]):
        marker = " <- optimal" if val == best_val else ""
        print(f"  Sequence {seq}: sumwiCi = {val}{marker}")
    
    print(f"\nOptimal: Sequence {best_seq}, sumwiCi = {best_val}")
    
    # Lower bound
    schedule = Schedule([j.clone() for j in jobs], '1', 1)
    lp_solver = LPRelaxationSolver(jobs, schedule, 'sumwiCi')
    lb_result = lp_solver.solve()
    print(f"Lower Bound: {lb_result['lower_bound']}")
    
    # WSPT should give optimal for 1||sumwiCi
    schedule2 = Schedule([j.clone() for j in jobs], '1', 1)
    DispatchingRules.WSPT(schedule2.jobs, schedule2)
    wspt_val = schedule2.get_objective_value('sumwiCi')
    print(f"WSPT sumwiCi: {wspt_val}")
    
    # Verify
    assert lb_result['lower_bound'] == best_val, "LB should equal optimal for 1||sumwiCi"
    assert wspt_val == best_val, "WSPT should give optimal for 1||sumwiCi"
    
    print(f"\n[PASS] TEST 4 PASSED: LB = Optimal = WSPT = {best_val}")
    sys.stdout.flush()
    return True


def test_flow_shop_cmax():
    """
    Test Case 5: Flow Shop (2 machines) | Cmax
    ==========================================
    
    Jobs (processing times on M1, M2):
    - Job 0: [4, 3]
    - Job 1: [2, 5]
    - Job 2: [3, 2]
    
    Johnson's rule for 2-machine flow shop:
    1. Find min processing time among all
    2. If it's on M1, schedule job first; if on M2, schedule last
    3. Repeat
    
    Min values: p(0,M2)=3, p(1,M1)=2, p(2,M2)=2
    - p(1,M1)=2 is on M1 -> Job 1 goes first
    - p(2,M2)=2 is on M2 -> Job 2 goes last
    - Job 0 goes in middle
    
    Optimal sequence: (1, 0, 2)
    
    Let's calculate (1, 0, 2):
    M1: Job1[0-2], Job0[2-6], Job2[6-9]
    M2: Job1[2-7], Job0[7-10], Job2[10-12]
    Cmax = 12
    """
    print("\n" + "=" * 60)
    print("TEST 5: Flow Shop (2 machines) | Cmax")
    print("=" * 60)
    
    jobs = [
        Job(0, [4, 3], None, 1, 0),
        Job(1, [2, 5], None, 1, 0),
        Job(2, [3, 2], None, 1, 0)
    ]
    
    # Brute force
    best_seq, best_val, all_results = brute_force_flow_shop(jobs, 2, 'Cmax')
    
    print("All permutations:")
    for seq, val in sorted(all_results, key=lambda x: x[1]):
        marker = " <- optimal" if val == best_val else ""
        print(f"  Sequence {seq}: Cmax = {val}{marker}")
    
    print(f"\nOptimal: Sequence {best_seq}, Cmax = {best_val}")
    
    # Lower bound
    schedule = Schedule([j.clone() for j in jobs], 'F', 2)
    lp_solver = LPRelaxationSolver(jobs, schedule, 'Cmax')
    lb_result = lp_solver.solve()
    print(f"Lower Bound: {lb_result['lower_bound']}")
    
    # LPT heuristic
    schedule2 = Schedule([j.clone() for j in jobs], 'F', 2)
    DispatchingRules.LPT(schedule2.jobs, schedule2)
    lpt_cmax = schedule2.get_objective_value('Cmax')
    print(f"LPT Cmax: {lpt_cmax}")
    
    # SPT heuristic  
    schedule3 = Schedule([j.clone() for j in jobs], 'F', 2)
    DispatchingRules.SPT(schedule3.jobs, schedule3)
    spt_cmax = schedule3.get_objective_value('Cmax')
    print(f"SPT Cmax: {spt_cmax}")
    
    # Verify lower bound is valid
    assert lb_result['lower_bound'] <= best_val, f"LB {lb_result['lower_bound']} should be <= optimal {best_val}"
    
    print(f"\n[PASS] TEST 5 PASSED: LB ({lb_result['lower_bound']}) <= Optimal ({best_val})")
    print(f"   LPT gives: {lpt_cmax}, SPT gives: {spt_cmax}")
    sys.stdout.flush()
    return True


def test_parallel_machines():
    """
    Test Case 6: Parallel Machines (2 machines) | Cmax
    ===================================================
    
    Jobs:
    - Job 0: p = 5
    - Job 1: p = 3
    - Job 2: p = 4
    - Job 3: p = 6
    
    Total processing time = 5+3+4+6 = 18
    LB = max(18/2, max(p)) = max(9, 6) = 9
    
    LPT order: (3,0,2,1) with p = (6,5,4,3)
    
    LPT scheduling:
    - Job3(p=6) -> M1: [0-6]
    - Job0(p=5) -> M2: [0-5]
    - Job2(p=4) -> M2: [5-9]  (M2 finishes at 5, earlier than M1 at 6)
    - Job1(p=3) -> M1: [6-9]  (M1 finishes at 6)
    
    Cmax = max(9, 9) = 9 = Optimal!
    """
    print("\n" + "=" * 60)
    print("TEST 6: Parallel Machines (2 machines) | Cmax")
    print("=" * 60)
    
    jobs = [
        Job(0, 5, None, 1, 0),
        Job(1, 3, None, 1, 0),
        Job(2, 4, None, 1, 0),
        Job(3, 6, None, 1, 0)
    ]
    
    total_p = sum(j.get_total_processing_time() for j in jobs)
    max_p = max(j.get_total_processing_time() for j in jobs)
    expected_lb = max(total_p / 2, max_p)
    
    print(f"Total processing time: {total_p}")
    print(f"Max processing time: {max_p}")
    print(f"Expected LB = max({total_p}/2, {max_p}) = {expected_lb}")
    
    # Lower bound
    schedule = Schedule([j.clone() for j in jobs], 'P', 2)
    lp_solver = LPRelaxationSolver(jobs, schedule, 'Cmax')
    lb_result = lp_solver.solve()
    print(f"Computed Lower Bound: {lb_result['lower_bound']}")
    
    # LPT heuristic
    schedule2 = Schedule([j.clone() for j in jobs], 'P', 2)
    DispatchingRules.LPT(schedule2.jobs, schedule2)
    lpt_cmax = schedule2.get_objective_value('Cmax')
    
    print(f"\nLPT Schedule:")
    for m in schedule2.machines:
        print(f"  Machine {m.id}: {[(t.job.id, t.start_time, t.end_time) for t in m.schedule]}")
    print(f"LPT Cmax: {lpt_cmax}")
    
    # Verify
    assert lb_result['lower_bound'] == expected_lb, f"LB should be {expected_lb}"
    assert lb_result['lower_bound'] <= lpt_cmax, "LB should be <= heuristic"
    
    print(f"\n[PASS] TEST 6 PASSED: LB = {lb_result['lower_bound']}, LPT = {lpt_cmax}")
    sys.stdout.flush()
    return True


def test_metaheuristic_convergence():
    """
    Test Case 7: Tabu Search Convergence
    =====================================
    
    Test that Tabu Search finds good solutions
    (at least as good as random, ideally close to optimal).
    """
    print("\n" + "=" * 60)
    print("TEST 7: Tabu Search Convergence")
    print("=" * 60)
    
    jobs = [
        Job(0, 5, None, 1, 0),
        Job(1, 3, None, 1, 0),
        Job(2, 7, None, 1, 0),
        Job(3, 2, None, 1, 0)
    ]
    
    # Known optimal for 1||sumCi is SPT order
    best_seq, best_val, _ = brute_force_single_machine(jobs, 'sumCi')
    print(f"Optimal sumCi: {best_val} (sequence {best_seq})")
    
    # Test TS
    schedule = Schedule([j.clone() for j in jobs], '1', 1)
    ts = TabuSearch(jobs, schedule, 'sumCi', {
        'tabu_list_size': 5,
        'max_iterations': 50
    })
    ts_result = ts.solve()
    print(f"TS result: {ts_result['fitness']}")
    
    # Verify Tabu Search found optimal (simple problem, it should)
    if ts_result['fitness'] == best_val:
        print(f"\n[PASS] TEST 7 PASSED: Tabu Search found optimal ({best_val})")
    else:
        print(f"\n[WARN] TEST 7: Tabu Search didn't find optimal")
        print(f"   This is acceptable for stochastic algorithms")
        assert ts_result['fitness'] <= best_val * 1.1, "TS should be within 10% of optimal"
        print("   Within 10% tolerance - OK")
    
    sys.stdout.flush()
    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print(" SCHEDULING DSS VERIFICATION TESTS")
    print("=" * 70)
    sys.stdout.flush()
    
    tests = [
        ("Single Machine | Cmax", test_single_machine_cmax),
        ("Single Machine | sumCi", test_single_machine_sumci),
        ("Single Machine | Lmax", test_single_machine_lmax),
        ("Single Machine | sumwiCi", test_single_machine_wspt),
        ("Flow Shop (2m) | Cmax", test_flow_shop_cmax),
        ("Parallel Machines (2m) | Cmax", test_parallel_machines),
        ("Tabu Search Convergence", test_metaheuristic_convergence),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
        except Exception as e:
            results.append((name, False, str(e)))
    
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed, error in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if error:
            print(f"         Error: {error}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL TESTS PASSED!")
    else:
        print(" SOME TESTS FAILED")
    print("=" * 70)
    sys.stdout.flush()
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()

