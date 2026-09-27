"""
Scheduling Decision Support System - Streamlit Application

Enhanced version with multiple constraint support and dynamic input fields
"""


import streamlit as st
import pandas as pd
import numpy as np
import time
from modules.scheduling_core import Job, Schedule
from modules.dispatching_rules import DispatchingRules
from modules.tabu_search import TabuSearch
from modules.genetic_algorithm import GeneticAlgorithm
from modules.simulated_annealing import SimulatedAnnealing
from modules.ant_colony import AntColonyOptimization
from modules.lp_solver import LPRelaxationSolver
from modules.visualization import create_gantt_chart, create_convergence_chart, create_comparison_chart

# Page configuration
st.set_page_config(
    page_title="Scheduling DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Scheduling Decision Support System")
st.markdown("*Advanced production scheduling with optimization algorithms*")
st.divider()

# Sidebar - Problem Definition
with st.sidebar:
    st.header("🔧 Problem Definition")
    
    # Alpha - Machine Environment
    alpha = st.selectbox(
        "α - Machine Environment",
        options=['', '1', 'P', 'F', 'J', 'O'],
        format_func=lambda x: {
            '': 'Select...',
            '1': '1 - Single Machine',
            'P': 'P - Parallel Machines',
            'F': 'F - Flow Shop',
            'J': 'J - Job Shop',
            'O': 'O - Open Shop'
        }[x]
    )
    
    # Beta - Constraints (MULTISELECT - KEY IMPROVEMENT!)
    beta = st.multiselect(
        "β - Constraints (can select multiple)",
        options=['prec', 'ri', 'di'],
        format_func=lambda x: {
            'prec': 'prec - Precedence',
            'ri': 'ri - Release Dates',
            'di': 'di - Deadlines'
        }[x],
        help="Select one or more constraints"
    )
    
    # Gamma - Objective
    gamma = st.selectbox(
        "γ - Objective Function",
        options=['', 'Cmax', 'sumCi', 'sumwiCi', 'Lmax', 'sumTi'],
        format_func=lambda x: {
            '': 'Select...',
            'Cmax': 'Cmax - Makespan',
            'sumCi': 'ΣCi - Total Completion Time',
            'sumwiCi': 'ΣwiCi - Weighted Completion',
            'Lmax': 'Lmax - Maximum Lateness',
            'sumTi': 'ΣTi - Total Tardiness'
        }[x]
    )
    
    # Display problem notation
    if alpha and gamma:
        beta_str = ','.join(beta) if beta else ''
        st.success(f"**Problem:** `{alpha}|{beta_str}|{gamma}`")

# Main area
if alpha and gamma:
    st.header("⚙️ Job Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_jobs = st.number_input("Number of Jobs", min_value=1, max_value=50, value=5)
    
    with col2:
        if alpha in ['P', 'F', 'J', 'O']:
            num_machines = st.number_input("Number of Machines", min_value=1, max_value=20, value=3)
        else:
            num_machines = 1
            st.info("Single machine problem")
    
    # Generate job table
    if st.button("🔄 Generate/Reset Job Table"):
        # Initialize session state for job data
        st.session_state.job_data = {}
        st.session_state.constraints_data = {}
        
        # Processing times
        # Single machine (1) and Parallel machines (P) have ONE processing time per job
        # Flow/Job/Open shop (F, J, O) have ONE processing time per machine
        if alpha in ['1', 'P']:
            st.session_state.job_data['Processing Time'] = np.random.randint(5, 25, num_jobs).tolist()
        else:
            # Flow, Job, Open Shop - need processing time for each machine
            for m in range(num_machines):
                st.session_state.job_data[f'M{m+1}'] = np.random.randint(5, 25, num_jobs).tolist()
        
        # Weights (always show for weighted objective)
        if gamma == 'sumwiCi':
            st.session_state.job_data['Weight'] = np.random.randint(1, 10, num_jobs).tolist()
        
        # Store constraint settings
        st.session_state.has_release_dates = 'ri' in beta
        st.session_state.has_due_dates = 'di' in beta or gamma in ['Lmax', 'sumTi']
        st.session_state.has_precedence = 'prec' in beta
        
        # Initialize release dates if constraint selected
        if st.session_state.has_release_dates:
            st.session_state.constraints_data['Release Date'] = np.random.randint(0, 20, num_jobs).tolist()
        
        # Initialize due dates if constraint selected or needed for objective
        if st.session_state.has_due_dates:
            st.session_state.constraints_data['Due Date'] = np.random.randint(50, 150, num_jobs).tolist()
        
        # Initialize precedence (empty by default)
        if st.session_state.has_precedence:
            st.session_state.constraints_data['Predecessors'] = [''] * num_jobs
        
        # Initialize job shop routing (random machine sequences)
        if alpha == 'J':
            routings = []
            for i in range(num_jobs):
                # Random permutation of machines (each job visits all machines in random order)
                route = list(np.random.permutation(num_machines))
                routings.append(','.join(str(m+1) for m in route))  # 1-indexed for display
            st.session_state.constraints_data['Routing'] = routings
    
    # Display and edit job table
    if 'job_data' in st.session_state:
        st.subheader("📋 Processing Times")
        
        df = pd.DataFrame(st.session_state.job_data)
        df.index = [f'Job {i+1}' for i in range(num_jobs)]
        
        # Editable dataframe for processing times
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            key="processing_times_editor"
        )
        
        # Release Dates Section
        if st.session_state.get('has_release_dates', False):
            st.subheader("📅 Release Dates (ri constraint)")
            st.caption("When each job becomes available for processing")
            
            release_df = pd.DataFrame({
                'Release Date': st.session_state.constraints_data.get('Release Date', [0] * num_jobs)
            })
            release_df.index = [f'Job {i+1}' for i in range(num_jobs)]
            
            edited_release = st.data_editor(
                release_df,
                use_container_width=True,
                num_rows="fixed",
                key="release_dates_editor"
            )
            st.session_state.constraints_data['Release Date'] = edited_release['Release Date'].tolist()
        
        # Due Dates Section  
        if st.session_state.get('has_due_dates', False):
            st.subheader("⏰ Due Dates (di constraint)")
            st.caption("Target completion times for calculating lateness/tardiness")
            
            due_df = pd.DataFrame({
                'Due Date': st.session_state.constraints_data.get('Due Date', [100] * num_jobs)
            })
            due_df.index = [f'Job {i+1}' for i in range(num_jobs)]
            
            edited_due = st.data_editor(
                due_df,
                use_container_width=True,
                num_rows="fixed",
                key="due_dates_editor"
            )
            st.session_state.constraints_data['Due Date'] = edited_due['Due Date'].tolist()
        
        # Precedence Constraints Section
        if st.session_state.get('has_precedence', False):
            st.subheader("🔗 Precedence Constraints (prec)")
            st.caption("Enter predecessor job numbers (comma-separated). Example: '1,3' means job must wait for Jobs 1 and 3")
            
            prec_df = pd.DataFrame({
                'Predecessors': st.session_state.constraints_data.get('Predecessors', [''] * num_jobs)
            })
            prec_df.index = [f'Job {i+1}' for i in range(num_jobs)]
            
            edited_prec = st.data_editor(
                prec_df,
                use_container_width=True,
                num_rows="fixed",
                key="precedence_editor"
            )
            st.session_state.constraints_data['Predecessors'] = edited_prec['Predecessors'].tolist()
        
        # Job Shop Routing Section
        if alpha == 'J':
            st.subheader("🛤️ Machine Routing (Job Shop)")
            st.caption(f"Enter machine visit order (comma-separated, 1-{num_machines}). Example: '2,1,3' means visit M2→M1→M3")
            
            routing_df = pd.DataFrame({
                'Routing': st.session_state.constraints_data.get('Routing', [','.join(str(m+1) for m in range(num_machines))] * num_jobs)
            })
            routing_df.index = [f'Job {i+1}' for i in range(num_jobs)]
            
            edited_routing = st.data_editor(
                routing_df,
                use_container_width=True,
                num_rows="fixed",
                key="routing_editor"
            )
            st.session_state.constraints_data['Routing'] = edited_routing['Routing'].tolist()
        
        st.divider()
        
        # Solver Selection
        st.header("🧮 Solver Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Dispatching Rules")
            use_spt = st.checkbox("SPT", help="Shortest Processing Time")
            use_lpt = st.checkbox("LPT", help="Longest Processing Time")
            use_edd = st.checkbox("EDD", help="Earliest Due Date")
            use_wspt = st.checkbox("WSPT", help="Weighted SPT")
            use_fifo = st.checkbox("FIFO", help="First In First Out")
        
        with col2:
            st.subheader("Metaheuristics")
            use_ga = st.checkbox("Genetic Algorithm")
            use_sa = st.checkbox("Simulated Annealing")
            use_ts = st.checkbox("Tabu Search")
            use_aco = st.checkbox("Ant Colony Optimization")
        
        with col3:
            st.subheader("Lower Bound")
            use_lp = st.checkbox("LP Relaxation")
        
        # Algorithm Parameters
        if any([use_ga, use_sa, use_ts, use_aco]):
            st.divider()
            st.header("🎛️ Algorithm Parameters")
            
            if use_ga:
                with st.expander("Genetic Algorithm Parameters", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        ga_pop = st.number_input("Population Size", 10, 200, 50, key="ga_pop")
                        ga_gen = st.number_input("Generations", 10, 1000, 100, key="ga_gen")
                    with col2:
                        ga_cx = st.slider("Crossover Rate", 0.0, 1.0, 0.8, key="ga_cx")
                        ga_mut = st.slider("Mutation Rate", 0.0, 1.0, 0.1, key="ga_mut")
            
            if use_sa:
                with st.expander("Simulated Annealing Parameters", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        sa_temp = st.number_input("Initial Temperature", 100, 10000, 1000, key="sa_temp")
                        sa_cool = st.slider("Cooling Rate", 0.8, 0.99, 0.95, key="sa_cool")
                    with col2:
                        sa_iter = st.number_input("Iterations per Temp", 10, 500, 100, key="sa_iter")
                        sa_min = st.number_input("Min Temperature", 0.1, 10.0, 1.0, key="sa_min")
            
            if use_ts:
                with st.expander("Tabu Search Parameters", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        ts_tabu = st.number_input("Tabu List Size", 5, 50, 10, key="ts_tabu")
                    with col2:
                        ts_iter = st.number_input("Max Iterations", 50, 10000, 200, key="ts_iter")
            
            if use_aco:
                with st.expander("Ant Colony Parameters", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        aco_ants = st.number_input("Number of Ants", 5, 100, 20, key="aco_ants")
                        aco_iter = st.number_input("Iterations", 20, 500, 100, key="aco_iter")
                    with col2:
                        aco_alpha = st.slider("Alpha (Pheromone)", 0.0, 5.0, 1.0, key="aco_alpha")
                        aco_beta = st.slider("Beta (Heuristic)", 0.0, 5.0, 2.0, key="aco_beta")
                        aco_evap = st.slider("Evaporation Rate", 0.0, 1.0, 0.5, key="aco_evap")
        
        st.divider()
        
        # Solve button
        if st.button("🚀 Solve Problem", type="primary", use_container_width=True):
            
            # Check if at least one solver is selected
            selected_solvers = [use_spt, use_lpt, use_edd, use_wspt, use_fifo, 
                              use_ga, use_sa, use_ts, use_aco, use_lp]
            
            if not any(selected_solvers):
                st.error("Please select at least one solver!")
            else:
                # Create jobs from dataframe
                jobs = []
                
                # Get constraint data
                constraints = st.session_state.get('constraints_data', {})
                
                # Validate dataframe columns match problem type
                try:
                    for i in range(num_jobs):
                        # Single machine and Parallel machines use ONE processing time
                        if alpha in ['1', 'P']:
                            p_times = edited_df.iloc[i]['Processing Time']
                        else:
                            # Flow/Job/Open Shop use list of processing times
                            p_times = [edited_df.iloc[i][f'M{m+1}'] for m in range(num_machines)]
                        
                        # Get weight from processing times editor
                        weight = edited_df.iloc[i].get('Weight', 1.0)
                        
                        # Get release date from constraints data
                        release_dates = constraints.get('Release Date', [])
                        release_date = release_dates[i] if i < len(release_dates) else 0.0
                        
                        # Get due date from constraints data
                        due_dates = constraints.get('Due Date', [])
                        due_date = due_dates[i] if i < len(due_dates) else None
                        
                        # Parse predecessors from constraints data
                        predecessors = []
                        prec_strs = constraints.get('Predecessors', [])
                        if i < len(prec_strs) and prec_strs[i]:
                            try:
                                # Parse comma-separated job numbers (1-indexed from UI)
                                for p in str(prec_strs[i]).split(','):
                                    p = p.strip()
                                    if p:
                                        pred_id = int(p) - 1  # Convert to 0-indexed
                                        if 0 <= pred_id < num_jobs and pred_id != i:
                                            predecessors.append(pred_id)
                            except ValueError:
                                pass  # Ignore invalid input
                        
                        # Parse routing for job shop
                        routing = []
                        if alpha == 'J':
                            routing_strs = constraints.get('Routing', [])
                            if i < len(routing_strs) and routing_strs[i]:
                                try:
                                    for m in str(routing_strs[i]).split(','):
                                        m = m.strip()
                                        if m:
                                            machine_id = int(m) - 1  # Convert to 0-indexed
                                            if 0 <= machine_id < num_machines:
                                                routing.append(machine_id)
                                except ValueError:
                                    pass  # Ignore invalid input
                        
                        job = Job(i, p_times, due_date, weight, release_date)
                        job.predecessors = predecessors
                        job.routing = routing
                        jobs.append(job)
                except KeyError as e:
                    st.error(f"⚠️ Column mismatch! Please click 'Generate/Reset Job Table' to create a new table for the selected problem type. Missing column: {e}")
                    st.stop()
                
                # Create schedule
                schedule = Schedule(jobs, alpha, num_machines)
                
                # Run solvers
                results = []
                lower_bound = None
                convergence_data = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                solver_list = []
                if use_spt: solver_list.append(('SPT', DispatchingRules.SPT))
                if use_lpt: solver_list.append(('LPT', DispatchingRules.LPT))
                if use_edd: solver_list.append(('EDD', DispatchingRules.EDD))
                if use_wspt: solver_list.append(('WSPT', DispatchingRules.WSPT))
                if use_fifo: solver_list.append(('FIFO', DispatchingRules.FIFO))
                
                total_solvers = len(solver_list) + sum([use_ga, use_sa, use_ts, use_aco, use_lp])
                current = 0
                
                # Dispatching rules
                for name, rule_func in solver_list:
                    status_text.text(f"Running {name}...")
                    start_time = time.time()
                    result_schedule = schedule.clone()
                    rule_func(result_schedule.jobs, result_schedule)
                    comp_time = time.time() - start_time
                    
                    results.append({
                        'solver_name': name,
                        'schedule': result_schedule,
                        'computation_time': comp_time * 1000,
                        'type': 'dispatching'
                    })
                    
                    current += 1
                    progress_bar.progress(current / total_solvers)
                
                # LP Solver
                if use_lp:
                    status_text.text("Computing lower bound...")
                    lp_solver = LPRelaxationSolver(jobs, schedule, gamma)
                    lp_result = lp_solver.solve()
                    lower_bound = lp_result['lower_bound']
                    
                    current += 1
                    progress_bar.progress(current / total_solvers)
                
                # Genetic Algorithm
                if use_ga:
                    status_text.text("Running Genetic Algorithm...")
                    ga = GeneticAlgorithm(jobs, schedule, gamma, {
                        'population_size': ga_pop,
                        'generations': ga_gen,
                        'crossover_rate': ga_cx,
                        'mutation_rate': ga_mut
                    })
                    start_time = time.time()
                    ga_result = ga.solve()
                    comp_time = time.time() - start_time
                    
                    results.append({
                        'solver_name': 'Genetic Algorithm',
                        'schedule': ga_result['schedule'],
                        'computation_time': comp_time * 1000,
                        'type': 'metaheuristic',
                        'convergence': ga_result['convergence']
                    })
                    convergence_data['GA'] = ga_result['convergence']
                    
                    current += 1
                    progress_bar.progress(current / total_solvers)
                
                # Simulated Annealing
                if use_sa:
                    status_text.text("Running Simulated Annealing...")
                    sa = SimulatedAnnealing(jobs, schedule, gamma, {
                        'initial_temp': sa_temp,
                        'cooling_rate': sa_cool,
                        'iterations_per_temp': sa_iter,
                        'min_temp': sa_min
                    })
                    start_time = time.time()
                    sa_result = sa.solve()
                    comp_time = time.time() - start_time
                    
                    results.append({
                        'solver_name': 'Simulated Annealing',
                        'schedule': sa_result['schedule'],
                        'computation_time': comp_time * 1000,
                        'type': 'metaheuristic',
                        'convergence': sa_result['convergence']
                    })
                    convergence_data['SA'] = sa_result['convergence']
                    
                    current += 1
                    progress_bar.progress(current / total_solvers)
                
                # Tabu Search
                if use_ts:
                    status_text.text("Running Tabu Search...")
                    ts = TabuSearch(jobs, schedule, gamma, {
                        'tabu_list_size': ts_tabu,
                        'max_iterations': ts_iter
                    })
                    start_time = time.time()
                    ts_result = ts.solve()
                    comp_time = time.time() - start_time
                    
                    results.append({
                        'solver_name': 'Tabu Search',
                        'schedule': ts_result['schedule'],
                        'computation_time': comp_time * 1000,
                        'type': 'metaheuristic',
                        'convergence': ts_result['convergence']
                    })
                    convergence_data['TS'] = ts_result['convergence']
                    
                    current += 1
                    progress_bar.progress(current / total_solvers)
                
                # Ant Colony Optimization
                if use_aco:
                    status_text.text("Running Ant Colony Optimization...")
                    aco = AntColonyOptimization(jobs, schedule, gamma, {
                        'num_ants': aco_ants,
                        'iterations': aco_iter,
                        'alpha': aco_alpha,
                        'beta': aco_beta,
                        'evaporation_rate': aco_evap
                    })
                    start_time = time.time()
                    aco_result = aco.solve()
                    comp_time = time.time() - start_time
                    
                    results.append({
                        'solver_name': 'Ant Colony',
                        'schedule': aco_result['schedule'],
                        'computation_time': comp_time * 1000,
                        'type': 'metaheuristic',
                        'convergence': aco_result['convergence']
                    })
                    convergence_data['ACO'] = aco_result['convergence']
                    
                    current += 1
                    progress_bar.progress(current / total_solvers)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display Results
                st.success("✅ Optimization Complete!")
                st.header("📊 Results")
                
                # Metrics cards
                cols = st.columns(min(len(results) + (1 if lower_bound else 0), 4))
                
                for idx, result in enumerate(results):
                    with cols[idx % len(cols)]:
                        obj_value = result['schedule'].get_objective_value(gamma)
                        st.metric(
                            label=result['solver_name'],
                            value=f"{obj_value:.2f}",
                            delta=f"{result['computation_time']:.0f}ms"
                        )
                        
                        if lower_bound and lower_bound > 0:
                            gap = ((obj_value - lower_bound) / lower_bound * 100)
                            st.caption(f"Gap: {gap:.1f}%")
                
                if lower_bound:
                    with cols[len(results) % len(cols)]:
                        st.metric(
                            label="Lower Bound",
                            value=f"{lower_bound:.2f}",
                            delta="LP"
                        )
                
                # Detailed table
                st.subheader("Detailed Metrics")
                metrics_data = []
                for result in results:
                    sch = result['schedule']
                    metrics_data.append({
                        'Solver': result['solver_name'],
                        'Makespan': f"{sch.makespan:.2f}",
                        'Total Completion': f"{sch.total_completion_time:.2f}",
                        'Max Lateness': f"{sch.max_lateness:.2f}" if sch.max_lateness != float('-inf') else 'N/A',
                        'Computation (ms)': f"{result['computation_time']:.0f}"
                    })
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                
                # Gantt Chart
                st.subheader("Schedule Visualization")
                best_result = min(results, key=lambda r: r['schedule'].get_objective_value(gamma))
                fig = create_gantt_chart(best_result['schedule'], f"Best Schedule - {best_result['solver_name']}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Convergence chart
                if convergence_data:
                    st.subheader("Algorithm Convergence")
                    conv_fig = create_convergence_chart(convergence_data)
                    st.plotly_chart(conv_fig, use_container_width=True)

else:
    st.info("👈 Please select problem type (α) and objective (γ) from the sidebar to begin")
    
    st.markdown("""
    ### How to Use
    
    1. **Define Problem**: Select machine environment (α), constraints (β), and objective (γ)
    2. **Configure Jobs**: Set number of jobs and machines, then generate the job table
    3. **Edit Values**: Modify processing times, release dates, due dates, and weights as needed
    4. **Select Solvers**: Choose one or more dispatching rules or metaheuristic algorithms
    5. **Tune Parameters**: Adjust algorithm parameters if using metaheuristics
    6. **Solve**: Click the solve button and view results!
    
    ### Features
    
    - ✅ **Multiple Constraints**: Select one or more constraints (pmtn, prec, ri, di)
    - ✅ **Dynamic Inputs**: Release and due date columns appear based on selection
    - ✅ **5 Dispatching Rules**: SPT, LPT, EDD, WSPT, FIFO
    - ✅ **4 Metaheuristics**: GA, SA, Tabu Search, Ant Colony
    - ✅ **LP Lower Bounds**: Compare solutions to theoretical optimum
    - ✅ **Interactive Visualization**: Gantt charts and convergence plots
    """)

# Footer
st.divider()
st.caption("Scheduling Decision Support System | Advanced Production Planning")
