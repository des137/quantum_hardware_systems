"""
IBM Miami Quantum Device Analysis - Main Module

This is the main entry point for analyzing IBM Miami calibration data.
It provides a high-level API for loading, analyzing, and visualizing the data.

Usage in Jupyter Notebook:
--------------------------
from ibm_miami_analysis import MiamiAnalyzer

analyzer = MiamiAnalyzer('ibm_miami_calibrations_2026-01-06T02_04_30Z.csv')
analyzer.summary()
analyzer.plot_all()

# Or individual analyses:
analyzer.analyze_coherence()
analyzer.analyze_cz_errors()
analyzer.analyze_readout()
analyzer.find_worst_qubits()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from ibm_miami_loader import (
    load_calibration_data,
    get_cz_errors_array,
    get_gate_lengths_array,
    extract_all_cz_errors,
    extract_all_gate_lengths,
    count_qubit_connections,
    identify_boundary_qubits,
    identify_interior_qubits,
    create_edge_dataframe,
    parse_cz_errors,
    parse_gate_lengths,
)

from ibm_miami_stats import (
    compute_stats_summary,
    detect_outliers_iqr,
    analyze_all_numeric_columns,
    get_outlier_qubits,
    get_cz_outlier_edges,
    correlation_analysis,
    t1_t2_ratio_analysis,
    readout_error_analysis,
    connection_statistics,
    generate_summary_report,
    StatsSummary,
)

from ibm_miami_viz import (
    set_plot_style,
    plot_histogram,
    plot_boxplot_comparison,
    plot_coherence_times,
    plot_cz_error_analysis,
    plot_readout_errors,
    plot_single_qubit_errors,
    plot_qubit_heatmap,
    plot_connectivity_graph,
    plot_outlier_summary,
    plot_correlation_matrix,
    create_full_report_figures,
)


class MiamiAnalyzer:
    """
    Main analyzer class for IBM Miami calibration data.

    This class provides a unified interface for loading, analyzing,
    and visualizing IBM Miami quantum device calibration data.

    Attributes
    ----------
    df : pd.DataFrame
        The loaded calibration data.
    edge_df : pd.DataFrame
        DataFrame with one row per CZ edge.
    n_qubits : int
        Number of qubits in the device.
    n_edges : int
        Number of CZ edges.
    """

    def __init__(self, csv_path: str):
        """
        Initialize analyzer with calibration CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the calibration CSV file.
        """
        self.csv_path = Path(csv_path)
        self.df = load_calibration_data(csv_path)
        self.edge_df = create_edge_dataframe(self.df)

        self.n_qubits = len(self.df)
        self.n_edges = len(self.edge_df)

        # Cache computed values
        self._cz_errors = None
        self._gate_lengths = None
        self._stats_cache = {}

        set_plot_style()
        print(f"Loaded IBM Miami calibration data: {self.n_qubits} qubits, {self.n_edges} CZ edges")

    @property
    def cz_errors(self) -> np.ndarray:
        """Get array of all CZ errors."""
        if self._cz_errors is None:
            self._cz_errors = get_cz_errors_array(self.df)
        return self._cz_errors

    @property
    def gate_lengths(self) -> np.ndarray:
        """Get array of all CZ gate lengths."""
        if self._gate_lengths is None:
            self._gate_lengths = get_gate_lengths_array(self.df)
        return self._gate_lengths

    def summary(self) -> str:
        """Print and return comprehensive summary report."""
        report = generate_summary_report(self.df)
        print(report)
        return report

    def get_column_stats(self, column: str) -> StatsSummary:
        """
        Get statistics for a specific column.

        Parameters
        ----------
        column : str
            Column name.

        Returns
        -------
        StatsSummary
            Statistics summary object.
        """
        if column not in self._stats_cache:
            data = self.df[column].dropna().values
            self._stats_cache[column] = compute_stats_summary(data, name=column)
        return self._stats_cache[column]

    def get_cz_stats(self) -> StatsSummary:
        """Get statistics for CZ errors."""
        return compute_stats_summary(self.cz_errors, name="CZ Error")

    def get_gate_length_stats(self) -> StatsSummary:
        """Get statistics for CZ gate lengths."""
        return compute_stats_summary(self.gate_lengths.astype(float), name="Gate Length (ns)")

    # =====================================================================
    # Analysis Methods
    # =====================================================================

    def analyze_coherence(self, show_plot: bool = True) -> Dict:
        """
        Analyze T1 and T2 coherence times.

        Parameters
        ----------
        show_plot : bool
            Whether to display plots.

        Returns
        -------
        dict
            Analysis results including statistics and outliers.
        """
        t1_stats = self.get_column_stats('T1 (us)')
        t2_stats = self.get_column_stats('T2 (us)')
        ratio_analysis = t1_t2_ratio_analysis(self.df)

        _, t1_low_outliers = get_outlier_qubits(self.df, 'T1 (us)')
        t2_low_outliers, _ = get_outlier_qubits(self.df, 'T2 (us)')

        result = {
            't1_stats': t1_stats,
            't2_stats': t2_stats,
            'ratio_analysis': ratio_analysis,
            't1_low_outliers': t1_low_outliers,
            't2_low_outliers': t2_low_outliers,
        }

        print("\n=== COHERENCE TIME ANALYSIS ===")
        print(t1_stats)
        print(t2_stats)
        print(f"\nT2/T1 Ratio: mean={ratio_analysis['mean_ratio']:.2f}, median={ratio_analysis['median_ratio']:.2f}")
        if ratio_analysis['violating_qubits']:
            print(f"WARNING: {len(ratio_analysis['violating_qubits'])} qubits violate T2 <= 2*T1 limit")

        if show_plot:
            plot_coherence_times(self.df)
            plt.show()

        return result

    def analyze_cz_errors(self, show_plot: bool = True) -> Dict:
        """
        Analyze CZ (two-qubit) gate errors.

        Parameters
        ----------
        show_plot : bool
            Whether to display plots.

        Returns
        -------
        dict
            Analysis results including statistics, outliers, and worst edges.
        """
        cz_stats = self.get_cz_stats()
        gl_stats = self.get_gate_length_stats()

        outlier_edges = get_cz_outlier_edges(self.df)
        worst_edges = self.edge_df.nlargest(10, 'cz_error')

        result = {
            'cz_stats': cz_stats,
            'gate_length_stats': gl_stats,
            'outlier_edges': outlier_edges,
            'worst_edges': worst_edges,
        }

        print("\n=== CZ ERROR ANALYSIS ===")
        print(cz_stats)
        print(gl_stats)
        print(f"\nOutlier edges (high error): {len(outlier_edges)}")
        print("\nTop 10 Worst CZ Edges:")
        print(worst_edges.to_string())

        if show_plot:
            plot_cz_error_analysis(self.df)
            plt.show()

        return result

    def analyze_readout(self, show_plot: bool = True) -> Dict:
        """
        Analyze readout/measurement errors.

        Parameters
        ----------
        show_plot : bool
            Whether to display plots.

        Returns
        -------
        dict
            Analysis results.
        """
        ro_stats = self.get_column_stats('Readout assignment error')
        ro_analysis = readout_error_analysis(self.df)

        _, high_outliers = get_outlier_qubits(self.df, 'Readout assignment error')
        worst_readout = self.df.nlargest(10, 'Readout assignment error')[
            ['Qubit', 'Readout assignment error', 'Prob meas0 prep1', 'Prob meas1 prep0']
        ]

        result = {
            'readout_stats': ro_stats,
            'readout_analysis': ro_analysis,
            'high_error_qubits': high_outliers,
            'worst_readout': worst_readout,
        }

        print("\n=== READOUT ERROR ANALYSIS ===")
        print(ro_stats)
        print(f"\nMean P(0|1): {ro_analysis['mean_p0_1']:.6f}")
        print(f"Mean P(1|0): {ro_analysis['mean_p1_0']:.6f}")
        print(f"Mean asymmetry: {ro_analysis['mean_asymmetry']:.6f}")
        print(f"\nQubits with high readout error: {high_outliers}")
        print("\nTop 10 Worst Readout Qubits:")
        print(worst_readout.to_string())

        if show_plot:
            plot_readout_errors(self.df)
            plt.show()

        return result

    def analyze_single_qubit_gates(self, show_plot: bool = True) -> Dict:
        """
        Analyze single-qubit gate errors.

        Parameters
        ----------
        show_plot : bool
            Whether to display plots.

        Returns
        -------
        dict
            Analysis results.
        """
        id_stats = self.get_column_stats('ID error')
        rx_stats = self.get_column_stats('RX error')

        _, high_id = get_outlier_qubits(self.df, 'ID error')
        worst_sq = self.df.nlargest(10, 'ID error')[
            ['Qubit', 'ID error', 'RX error', '√x (sx) error', 'Pauli-X error']
        ]

        result = {
            'id_error_stats': id_stats,
            'rx_error_stats': rx_stats,
            'high_error_qubits': high_id,
            'worst_single_qubit': worst_sq,
        }

        print("\n=== SINGLE-QUBIT GATE ERROR ANALYSIS ===")
        print(id_stats)
        print(f"\nQubits with high single-qubit error: {high_id}")
        print("\nTop 10 Worst Single-Qubit Error Qubits:")
        print(worst_sq.to_string())

        if show_plot:
            plot_single_qubit_errors(self.df)
            plt.show()

        return result

    def analyze_connectivity(self, show_plot: bool = True) -> Dict:
        """
        Analyze device connectivity.

        Parameters
        ----------
        show_plot : bool
            Whether to display plots.

        Returns
        -------
        dict
            Connectivity analysis results.
        """
        conn_stats = connection_statistics(self.df)
        boundary = identify_boundary_qubits(self.df)
        interior = identify_interior_qubits(self.df)

        result = {
            'connection_stats': conn_stats,
            'boundary_qubits': boundary,
            'interior_qubits': interior,
        }

        print("\n=== CONNECTIVITY ANALYSIS ===")
        print(f"Total qubits: {conn_stats['total_qubits']}")
        print(f"Mean connections: {conn_stats['mean_connections']:.2f}")
        print(f"Connection distribution: {conn_stats['connection_distribution']}")
        print(f"Boundary qubits ({len(boundary)}): {boundary}")
        print(f"Interior qubits ({len(interior)}): {len(interior)} qubits with 4 connections")

        if show_plot:
            plot_connectivity_graph(self.df)
            plt.show()

        return result

    def find_worst_qubits(self, n: int = 10) -> pd.DataFrame:
        """
        Find the worst performing qubits across multiple metrics.

        Parameters
        ----------
        n : int
            Number of worst qubits to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with worst performing qubits.
        """
        # Create a composite score (higher = worse)
        df = self.df.copy()

        # Normalize each metric
        metrics = ['Readout assignment error', 'ID error', 'MEASURE error']

        # Add average CZ error for each qubit
        avg_cz = {}
        for _, row in df.iterrows():
            qubit = row['Qubit']
            cz_dict = parse_cz_errors(row['CZ error'])
            if cz_dict:
                avg_cz[qubit] = np.mean(list(cz_dict.values()))
            else:
                avg_cz[qubit] = 0
        df['avg_cz_error'] = df['Qubit'].map(avg_cz)

        # Inverse metrics (low is bad)
        df['T1_inverse'] = 1 / (df['T1 (us)'] + 1e-10)
        df['T2_inverse'] = 1 / (df['T2 (us)'] + 1e-10)

        # Z-score normalization for composite score
        from scipy import stats as scipy_stats

        score_metrics = metrics + ['avg_cz_error', 'T1_inverse', 'T2_inverse']
        for m in score_metrics:
            df[f'{m}_zscore'] = scipy_stats.zscore(df[m])

        df['composite_score'] = sum(df[f'{m}_zscore'] for m in score_metrics)

        worst = df.nlargest(n, 'composite_score')[
            ['Qubit', 'T1 (us)', 'T2 (us)', 'Readout assignment error',
             'ID error', 'avg_cz_error', 'composite_score']
        ]

        print(f"\n=== TOP {n} WORST PERFORMING QUBITS ===")
        print("(Based on composite score of T1, T2, readout, ID, and CZ errors)")
        print(worst.to_string())

        return worst

    def find_best_qubits(self, n: int = 10) -> pd.DataFrame:
        """
        Find the best performing qubits across multiple metrics.

        Parameters
        ----------
        n : int
            Number of best qubits to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with best performing qubits.
        """
        df = self.df.copy()

        # Add average CZ error
        avg_cz = {}
        for _, row in df.iterrows():
            qubit = row['Qubit']
            cz_dict = parse_cz_errors(row['CZ error'])
            if cz_dict:
                avg_cz[qubit] = np.mean(list(cz_dict.values()))
            else:
                avg_cz[qubit] = 0
        df['avg_cz_error'] = df['Qubit'].map(avg_cz)

        # Inverse metrics (low is bad, so high inverse is bad)
        df['T1_inverse'] = 1 / (df['T1 (us)'] + 1e-10)
        df['T2_inverse'] = 1 / (df['T2 (us)'] + 1e-10)

        from scipy import stats as scipy_stats

        metrics = ['Readout assignment error', 'ID error', 'MEASURE error']
        score_metrics = metrics + ['avg_cz_error', 'T1_inverse', 'T2_inverse']

        for m in score_metrics:
            df[f'{m}_zscore'] = scipy_stats.zscore(df[m])

        df['composite_score'] = sum(df[f'{m}_zscore'] for m in score_metrics)

        best = df.nsmallest(n, 'composite_score')[
            ['Qubit', 'T1 (us)', 'T2 (us)', 'Readout assignment error',
             'ID error', 'avg_cz_error', 'composite_score']
        ]

        print(f"\n=== TOP {n} BEST PERFORMING QUBITS ===")
        print("(Based on composite score of T1, T2, readout, ID, and CZ errors)")
        print(best.to_string())

        return best

    # =====================================================================
    # Plotting Methods
    # =====================================================================

    def plot_all(self, save_prefix: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create all analysis plots.

        Parameters
        ----------
        save_prefix : str, optional
            If provided, save figures with this prefix.

        Returns
        -------
        dict
            Dictionary of figure names to Figure objects.
        """
        return create_full_report_figures(self.df, save_prefix)

    def plot_metric(
        self,
        column: str,
        log_scale: bool = False,
        plot_type: str = 'histogram'
    ) -> plt.Figure:
        """
        Plot a specific metric.

        Parameters
        ----------
        column : str
            Column name to plot.
        log_scale : bool
            Use log scale.
        plot_type : str
            'histogram', 'bar', or 'heatmap'

        Returns
        -------
        plt.Figure
            The figure object.
        """
        if plot_type == 'histogram':
            fig, ax = plt.subplots(figsize=(10, 6))
            data = self.df[column].values
            plot_histogram(data, column, column, log_scale=log_scale, ax=ax)
        elif plot_type == 'bar':
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(self.df['Qubit'], self.df[column], alpha=0.7)
            ax.set_xlabel('Qubit')
            ax.set_ylabel(column)
            ax.set_title(f'{column} by Qubit')
            if log_scale:
                ax.set_yscale('log')
        elif plot_type == 'heatmap':
            fig = plot_qubit_heatmap(self.df, column, log_scale=log_scale)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        return fig

    def plot_correlation(self) -> plt.Figure:
        """Plot correlation matrix."""
        return plot_correlation_matrix(self.df)

    def plot_connectivity(self, color_by: str = 'cz_error') -> plt.Figure:
        """
        Plot device connectivity graph.

        Parameters
        ----------
        color_by : str
            'cz_error' or 'gate_length'
        """
        return plot_connectivity_graph(self.df, color_by=color_by)

    # =====================================================================
    # Data Access Methods
    # =====================================================================

    def get_qubit_data(self, qubit: int) -> pd.Series:
        """Get all data for a specific qubit."""
        return self.df[self.df['Qubit'] == qubit].iloc[0]

    def get_edge_data(self, qubit1: int, qubit2: int) -> Optional[pd.Series]:
        """Get CZ edge data between two qubits."""
        q1, q2 = min(qubit1, qubit2), max(qubit1, qubit2)
        mask = (self.edge_df['qubit1'] == q1) & (self.edge_df['qubit2'] == q2)
        if mask.any():
            return self.edge_df[mask].iloc[0]
        return None

    def get_neighbors(self, qubit: int) -> List[int]:
        """Get list of neighboring qubits (connected via CZ)."""
        row = self.df[self.df['Qubit'] == qubit].iloc[0]
        cz_dict = parse_cz_errors(row['CZ error'])
        return list(cz_dict.keys())

    def filter_qubits(
        self,
        min_t1: Optional[float] = None,
        max_readout_error: Optional[float] = None,
        max_id_error: Optional[float] = None,
    ) -> List[int]:
        """
        Filter qubits based on quality criteria.

        Parameters
        ----------
        min_t1 : float, optional
            Minimum T1 time in microseconds.
        max_readout_error : float, optional
            Maximum readout error.
        max_id_error : float, optional
            Maximum ID error.

        Returns
        -------
        list
            List of qubit numbers meeting all criteria.
        """
        mask = pd.Series([True] * len(self.df))

        if min_t1 is not None:
            mask &= self.df['T1 (us)'] >= min_t1
        if max_readout_error is not None:
            mask &= self.df['Readout assignment error'] <= max_readout_error
        if max_id_error is not None:
            mask &= self.df['ID error'] <= max_id_error

        return self.df[mask]['Qubit'].tolist()


def quick_analysis(csv_path: str) -> MiamiAnalyzer:
    """
    Run a quick analysis and return the analyzer object.

    Parameters
    ----------
    csv_path : str
        Path to calibration CSV.

    Returns
    -------
    MiamiAnalyzer
        Configured analyzer object.
    """
    analyzer = MiamiAnalyzer(csv_path)
    analyzer.summary()
    return analyzer


if __name__ == "__main__":
    import glob
    import sys

    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found in current directory")
        sys.exit(1)

    csv_path = csv_files[0]
    print(f"Analyzing: {csv_path}\n")

    analyzer = MiamiAnalyzer(csv_path)

    # Run all analyses
    analyzer.summary()
    print("\n" + "="*70 + "\n")

    analyzer.analyze_coherence(show_plot=False)
    analyzer.analyze_cz_errors(show_plot=False)
    analyzer.analyze_readout(show_plot=False)
    analyzer.analyze_single_qubit_gates(show_plot=False)
    analyzer.analyze_connectivity(show_plot=False)

    print("\n" + "="*70)
    analyzer.find_worst_qubits()
    analyzer.find_best_qubits()

    print("\n\nTo generate plots, run:")
    print("  analyzer.plot_all()")
    print("  plt.show()")
