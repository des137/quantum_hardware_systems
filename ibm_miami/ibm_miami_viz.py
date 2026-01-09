"""
IBM Miami Quantum Device Visualization

This module provides visualization functions for IBM Miami calibration data,
including histograms, box plots, heatmaps, and device topology visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats

from ibm_miami_loader import (
    load_calibration_data,
    get_cz_errors_array,
    get_gate_lengths_array,
    extract_all_cz_errors,
    extract_all_gate_lengths,
    count_qubit_connections,
    create_edge_dataframe,
    parse_cz_errors,
)
from ibm_miami_stats import (
    compute_stats_summary,
    detect_outliers_iqr,
)


def set_plot_style():
    """Set consistent plot style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_histogram(
    data: np.ndarray,
    title: str,
    xlabel: str,
    log_scale: bool = False,
    bins: int = 50,
    show_stats: bool = True,
    highlight_outliers: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot histogram with optional log scale and outlier highlighting.

    Parameters
    ----------
    data : np.ndarray
        Data to plot.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    log_scale : bool
        Use logarithmic scale for x-axis.
    bins : int
        Number of histogram bins.
    show_stats : bool
        Show statistics annotation.
    highlight_outliers : bool
        Highlight outlier regions.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if log_scale:
        # Use log bins
        data_positive = data[data > 0]
        log_bins = np.logspace(
            np.log10(data_positive.min()),
            np.log10(data_positive.max()),
            bins
        )
        ax.hist(data_positive, bins=log_bins, edgecolor='black', alpha=0.7)
        ax.set_xscale('log')
    else:
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)

    if highlight_outliers:
        mask, outliers, low, high = detect_outliers_iqr(data)
        if not log_scale:
            ax.axvline(low, color='red', linestyle='--', alpha=0.7, label=f'Low threshold: {low:.4f}')
            ax.axvline(high, color='red', linestyle='--', alpha=0.7, label=f'High threshold: {high:.4f}')
        else:
            if high > 0:
                ax.axvline(high, color='red', linestyle='--', alpha=0.7, label=f'High threshold: {high:.4f}')

    if show_stats:
        stats_text = f"n={len(data)}\nmean={np.mean(data):.4f}\nmedian={np.median(data):.4f}\nstd={np.std(data):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    if highlight_outliers:
        ax.legend(loc='upper left')

    return ax


def plot_boxplot_comparison(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Parameter Comparison",
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot box plots comparing multiple columns.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    columns : list
        Column names to compare.
    title : str
        Plot title.
    log_scale : bool
        Use log scale for y-axis.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    data = [df[col].dropna().values for col in columns]
    bp = ax.boxplot(data, labels=columns, patch_artist=True)

    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    if log_scale:
        ax.set_yscale('log')

    ax.set_title(title)
    ax.set_ylabel('Value')
    plt.xticks(rotation=45, ha='right')

    return ax


def plot_coherence_times(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot comprehensive coherence time analysis (T1, T2).

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    t1 = df['T1 (us)'].values
    t2 = df['T2 (us)'].values

    # T1 histogram
    plot_histogram(t1, 'T1 Distribution', 'T1 (μs)', ax=axes[0, 0])

    # T2 histogram
    plot_histogram(t2, 'T2 Distribution', 'T2 (μs)', ax=axes[0, 1])

    # T1 vs T2 scatter
    ax = axes[1, 0]
    ax.scatter(t1, t2, alpha=0.6, c=df['Qubit'], cmap='viridis')
    ax.plot([0, max(t1)], [0, 2*max(t1)], 'r--', alpha=0.5, label='T2 = 2*T1 limit')
    ax.set_xlabel('T1 (μs)')
    ax.set_ylabel('T2 (μs)')
    ax.set_title('T1 vs T2 (color = qubit number)')
    ax.legend()

    # T2/T1 ratio histogram
    ratio = t2 / t1
    ax = axes[1, 1]
    ax.hist(ratio, bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(2.0, color='red', linestyle='--', label='T2 = 2*T1 limit')
    ax.axvline(np.median(ratio), color='green', linestyle='-', label=f'Median: {np.median(ratio):.2f}')
    ax.set_xlabel('T2/T1 Ratio')
    ax.set_ylabel('Count')
    ax.set_title('T2/T1 Ratio Distribution')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_cz_error_analysis(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot comprehensive CZ error analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    cz_errors = get_cz_errors_array(df)
    gate_lengths = get_gate_lengths_array(df)

    # CZ error histogram (linear)
    plot_histogram(cz_errors, 'CZ Error Distribution', 'CZ Error',
                   highlight_outliers=True, ax=axes[0, 0])

    # CZ error histogram (log scale) - better for seeing distribution
    plot_histogram(cz_errors, 'CZ Error Distribution (Log Scale)', 'CZ Error',
                   log_scale=True, ax=axes[0, 1])

    # Gate length histogram
    ax = axes[1, 0]
    ax.hist(gate_lengths, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Gate Length (ns)')
    ax.set_ylabel('Count')
    ax.set_title('CZ Gate Length Distribution')

    # CZ error vs gate length
    edge_df = create_edge_dataframe(df)
    ax = axes[1, 1]
    sc = ax.scatter(edge_df['gate_length_ns'], edge_df['cz_error'],
                    alpha=0.5, c=range(len(edge_df)), cmap='viridis')
    ax.set_xlabel('Gate Length (ns)')
    ax.set_ylabel('CZ Error')
    ax.set_title('CZ Error vs Gate Length')
    ax.set_yscale('log')

    plt.tight_layout()
    return fig


def plot_readout_errors(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot comprehensive readout error analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Readout assignment error histogram
    plot_histogram(
        df['Readout assignment error'].values,
        'Readout Assignment Error',
        'Error',
        ax=axes[0, 0]
    )

    # P(0|1) vs P(1|0) scatter
    ax = axes[0, 1]
    ax.scatter(df['Prob meas0 prep1'], df['Prob meas1 prep0'],
               alpha=0.6, c=df['Qubit'], cmap='viridis')
    ax.plot([0, 0.25], [0, 0.25], 'r--', alpha=0.5, label='Equal errors')
    ax.set_xlabel('P(0|1) - Prepared 1, Measured 0')
    ax.set_ylabel('P(1|0) - Prepared 0, Measured 1')
    ax.set_title('Readout Error Asymmetry')
    ax.legend()

    # Readout error by qubit
    ax = axes[1, 0]
    ax.bar(df['Qubit'], df['Readout assignment error'], alpha=0.7)
    ax.set_xlabel('Qubit')
    ax.set_ylabel('Readout Assignment Error')
    ax.set_title('Readout Error by Qubit')

    # MEASURE error histogram (log scale helpful for outliers)
    plot_histogram(
        df['MEASURE error'].values,
        'MEASURE Error Distribution',
        'MEASURE Error',
        log_scale=True,
        ax=axes[1, 1]
    )

    plt.tight_layout()
    return fig


def plot_single_qubit_errors(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot single-qubit gate error analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ID error histogram (log scale)
    plot_histogram(
        df['ID error'].values,
        'ID Error Distribution',
        'ID Error',
        log_scale=True,
        ax=axes[0, 0]
    )

    # RX error histogram
    plot_histogram(
        df['RX error'].values,
        'RX Error Distribution',
        'RX Error',
        log_scale=True,
        ax=axes[0, 1]
    )

    # ID error by qubit
    ax = axes[1, 0]
    ax.bar(df['Qubit'], df['ID error'], alpha=0.7)
    ax.set_xlabel('Qubit')
    ax.set_ylabel('ID Error')
    ax.set_title('ID Error by Qubit')
    ax.set_yscale('log')

    # Comparison boxplot
    error_cols = ['ID error', 'RX error', '√x (sx) error', 'Pauli-X error']
    plot_boxplot_comparison(df, error_cols, 'Single-Qubit Gate Errors Comparison',
                           log_scale=True, ax=axes[1, 1])

    plt.tight_layout()
    return fig


def plot_qubit_heatmap(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    log_scale: bool = False,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot a heatmap of qubit values arranged in device topology.

    Note: This assumes a roughly rectangular grid arrangement.
    For actual IBM Miami topology, positions would need adjustment.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    column : str
        Column to visualize.
    title : str, optional
        Plot title.
    cmap : str
        Colormap name.
    log_scale : bool
        Use log scale for colors.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # IBM Miami is a 12x10 heavy-hex lattice (approximately)
    # Let's arrange qubits in a grid pattern for visualization
    n_qubits = len(df)
    n_cols = 12
    n_rows = int(np.ceil(n_qubits / n_cols))

    values = df[column].values
    if log_scale:
        values = np.log10(values + 1e-10)

    # Create grid
    grid = np.full((n_rows, n_cols), np.nan)
    for i, val in enumerate(values):
        row = i // n_cols
        col = i % n_cols
        grid[row, col] = val

    im = ax.imshow(grid, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label=column + (' (log10)' if log_scale else ''))

    # Add qubit labels
    for i in range(n_qubits):
        row = i // n_cols
        col = i % n_cols
        ax.text(col, row, str(i), ha='center', va='center', fontsize=7, color='white')

    ax.set_title(title or f'{column} by Qubit Position')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    return fig


def plot_connectivity_graph(
    df: pd.DataFrame,
    color_by: str = 'cz_error',
    figsize: Tuple[int, int] = (14, 12),
) -> plt.Figure:
    """
    Plot device connectivity graph with edges colored by CZ error.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    color_by : str
        'cz_error' or 'gate_length'
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Arrange qubits in grid
    n_qubits = len(df)
    n_cols = 12
    positions = {}
    for i in range(n_qubits):
        row = i // n_cols
        col = i % n_cols
        positions[i] = (col * 1.5, -row * 1.5)

    edge_df = create_edge_dataframe(df)

    # Get color values
    if color_by == 'cz_error':
        colors = edge_df['cz_error'].values
        cmap = 'Reds'
        label = 'CZ Error'
    else:
        colors = edge_df['gate_length_ns'].values
        cmap = 'Blues'
        label = 'Gate Length (ns)'

    # Normalize colors
    norm = mcolors.LogNorm(vmin=colors.min(), vmax=colors.max()) if color_by == 'cz_error' else mcolors.Normalize(vmin=colors.min(), vmax=colors.max())

    # Draw edges
    segments = []
    edge_colors = []
    for idx, row in edge_df.iterrows():
        q1, q2 = int(row['qubit1']), int(row['qubit2'])
        if q1 in positions and q2 in positions:
            segments.append([positions[q1], positions[q2]])
            edge_colors.append(row[color_by.replace('_', '_')] if color_by == 'cz_error' else row['gate_length_ns'])

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2)
    lc.set_array(np.array(edge_colors))
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label=label)

    # Draw nodes
    connections = count_qubit_connections(df)
    for qubit, pos in positions.items():
        n_conn = connections.get(qubit, 0)
        color = 'lightblue' if n_conn == 4 else 'lightcoral'  # Interior vs boundary
        circle = Circle(pos, 0.4, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], str(qubit), ha='center', va='center', fontsize=7)

    ax.set_xlim(-1, n_cols * 1.5)
    ax.set_ylim(-((n_qubits // n_cols) + 1) * 1.5, 1)
    ax.set_aspect('equal')
    ax.set_title(f'Device Connectivity (edges colored by {label})')
    ax.axis('off')

    # Legend for node colors
    ax.plot([], [], 'o', color='lightblue', markersize=10, label='Interior (4 conn)')
    ax.plot([], [], 'o', color='lightcoral', markersize=10, label='Boundary (<4 conn)')
    ax.legend(loc='upper right')

    return fig


def plot_outlier_summary(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Create a summary plot of outliers across all metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    metrics = [
        ('T1 (us)', 'T1 (μs)', False),
        ('T2 (us)', 'T2 (μs)', False),
        ('Readout assignment error', 'Readout Error', False),
        ('ID error', 'ID Error', True),
        ('MEASURE error', 'MEASURE Error', True),
    ]

    # CZ errors
    cz_errors = get_cz_errors_array(df)
    ax = axes[0]
    plot_histogram(cz_errors, 'CZ Error (Log)', 'CZ Error', log_scale=True, ax=ax)

    for i, (col, label, use_log) in enumerate(metrics):
        ax = axes[i + 1]
        data = df[col].values
        plot_histogram(data, label, label, log_scale=use_log, ax=ax)

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot correlation matrix heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    numeric_cols = [
        'T1 (us)', 'T2 (us)', 'Readout assignment error',
        'Prob meas0 prep1', 'Prob meas1 prep0', 'ID error', 'MEASURE error'
    ]
    existing_cols = [c for c in numeric_cols if c in df.columns]

    corr = df[existing_cols].corr()

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')

    # Labels
    ax.set_xticks(range(len(existing_cols)))
    ax.set_yticks(range(len(existing_cols)))
    ax.set_xticklabels(existing_cols, rotation=45, ha='right')
    ax.set_yticklabels(existing_cols)

    # Annotate
    for i in range(len(existing_cols)):
        for j in range(len(existing_cols)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)

    ax.set_title('Parameter Correlation Matrix')
    plt.tight_layout()

    return fig


def create_full_report_figures(
    df: pd.DataFrame,
    save_prefix: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Create all analysis figures.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    save_prefix : str, optional
        If provided, save figures with this prefix.

    Returns
    -------
    dict
        Dictionary of figure names to Figure objects.
    """
    set_plot_style()

    figures = {}

    print("Creating coherence time plots...")
    figures['coherence'] = plot_coherence_times(df)

    print("Creating CZ error plots...")
    figures['cz_errors'] = plot_cz_error_analysis(df)

    print("Creating readout error plots...")
    figures['readout'] = plot_readout_errors(df)

    print("Creating single-qubit error plots...")
    figures['single_qubit'] = plot_single_qubit_errors(df)

    print("Creating outlier summary...")
    figures['outliers'] = plot_outlier_summary(df)

    print("Creating correlation matrix...")
    figures['correlation'] = plot_correlation_matrix(df)

    print("Creating connectivity graph...")
    figures['connectivity'] = plot_connectivity_graph(df)

    if save_prefix:
        for name, fig in figures.items():
            filepath = f"{save_prefix}_{name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")

    return figures


if __name__ == "__main__":
    import glob

    csv_files = glob.glob("*.csv")
    if csv_files:
        df = load_calibration_data(csv_files[0])
        figures = create_full_report_figures(df, save_prefix="ibm_miami")
        plt.show()
