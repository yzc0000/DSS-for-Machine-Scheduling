"""
Visualization module using Plotly
"""
import plotly.figure_factory as ff
import plotly.graph_objects as go
from typing import List
import pandas as pd
from modules.scheduling_core import Schedule


def create_gantt_chart(schedule: Schedule, title: str = "Schedule") -> go.Figure:
    """Create interactive Gantt chart using Plotly with numeric time axis"""
    
    # Prepare data for Gantt chart
    colors = [
        '#60A5FA', '#34D399', '#F472B6', '#FBBF24', '#A78BFA',
        '#FB923C', '#4ADE80', '#38BDF8', '#F87171', '#C084FC'
    ]
    
    if not any(machine.schedule for machine in schedule.machines):
        # Empty schedule
        fig = go.Figure()
        fig.add_annotation(
            text="No schedule to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # Create horizontal bars for each task
    for machine in schedule.machines:
        for task in machine.schedule:
            job_color = colors[task.job.id % len(colors)]
            duration = task.end_time - task.start_time
            
            fig.add_trace(go.Bar(
                x=[duration],
                y=[f'Machine {machine.id + 1}'],
                base=[task.start_time],
                orientation='h',
                name=f'Job {task.job.id + 1}',
                marker=dict(color=job_color),
                text=f'Job {task.job.id + 1} (p={duration:.1f})',
                textposition='inside',
                insidetextanchor='middle',
                hovertemplate=f'Job {task.job.id + 1}<br>Start: {task.start_time:.1f}<br>End: {task.end_time:.1f}<br>Duration: {duration:.1f}<extra></extra>',
                showlegend=False
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Machine",
        barmode='overlay',
        height=400,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )
    
    return fig


def create_convergence_chart(convergence_data: dict, title: str = "Algorithm Convergence") -> go.Figure:
    """Create convergence plot for metaheuristics"""
    
    fig = go.Figure()
    
    for name, convergence in convergence_data.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(convergence))),
            y=convergence,
            mode='lines',
            name=name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Objective Value",
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_comparison_chart(results: List[dict], objective: str) -> go.Figure:
    """Create bar chart comparing solver results"""
    
    solver_names = []
    objective_values = []
    computation_times = []
    
    for result in results:
        if 'schedule' in result:
            solver_names.append(result['solver_name'])
            objective_values.append(result['schedule'].get_objective_value(objective))
            computation_times.append(result.get('computation_time', 0))
    
    fig = go.Figure(data=[
        go.Bar(
            name='Objective Value',
            x=solver_names,
            y=objective_values,
            marker_color='#60A5FA'
        )
    ])
    
    fig.update_layout(
        title="Solver Comparison",
        xaxis_title="Solver",
        yaxis_title=f"Objective Value ({objective})",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
