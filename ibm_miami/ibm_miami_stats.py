"""
IBM Miami Quantum Device Statistical Analysis

This module provides statistical analysis functions for IBM Miami calibration data,
including outlier detection, distribution analysis, and summary statistics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from ibm_miami_loader import (
    load_calibration_data,
    get_cz_errors_array,
    get_gate_lengths_array,
    count_qubit_connections,
    create_edge_dataframe,
)


@dataclass
class StatsSummary:
    """Container for statistical summary of a dataset."""
    name: str
    count: int
    mean: float
    std: float
    median: float
    q1: float
    q3: float
    iqr: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float
    outlier_count: int
    outlier_threshold_low: float
    outlier_threshold_high: float

    def __str__(self) -> str:
        return f"""
{self.name} Statistics:
  Count: {self.count}
  Mean: {self.mean:.6f}
  Std: {self.std:.6f}
  Median: {self.median:.6f}
  Q1 (25%): {self.q1:.6f}
  Q3 (75%): {self.q3:.6f}
  IQR: {self.iqr:.6f}
  Min: {self.min_val:.6f}
  Max: {self.max_val:.6f}
  Skewness: {self.skewness:.4f}
  Kurtosis: {self.kurtosis:.4f}
  Outliers: {self.outlier_count} (threshold: < {self.outlier_threshold_low:.6f} or > {self.outlier_threshold_high:.6f})
"""


def compute_stats_summary(data: np.ndarray, name: str = "Data", iqr_factor: float = 1.5) -> StatsSummary:
    """
    Compute comprehensive statistics summary for a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    name : str
        Name for the dataset.
    iqr_factor : float
        Factor for IQR-based outlier detection (default 1.5).

    Returns
    -------
    StatsSummary
        Dataclass containing all statistics.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    low_threshold = q1 - iqr_factor * iqr
    high_threshold = q3 + iqr_factor * iqr

    outliers = (data < low_threshold) | (data > high_threshold)

    return StatsSummary(
        name=name,
        count=len(data),
        mean=np.mean(data),
        std=np.std(data),
        median=np.median(data),
        q1=q1,
        q3=q3,
        iqr=iqr,
        min_val=np.min(data),
        max_val=np.max(data),
        skewness=stats.skew(data),
        kurtosis=stats.kurtosis(data),
        outlier_count=np.sum(outliers),
        outlier_threshold_low=low_threshold,
        outlier_threshold_high=high_threshold,
    )


def detect_outliers_iqr(
    data: np.ndarray, factor: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Detect outliers using IQR method.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    factor : float
        IQR multiplier for outlier thresholds.

    Returns
    -------
    tuple
        (outlier_mask, outlier_values, low_threshold, high_threshold)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    low = q1 - factor * iqr
    high = q3 + factor * iqr

    mask = (data < low) | (data > high)
    return mask, data[mask], low, high


def detect_outliers_zscore(
    data: np.ndarray, threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using Z-score method.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    threshold : float
        Z-score threshold for outliers.

    Returns
    -------
    tuple
        (outlier_mask, outlier_values)
    """
    z_scores = np.abs(stats.zscore(data))
    mask = z_scores > threshold
    return mask, data[mask]


def analyze_column(df: pd.DataFrame, column: str) -> StatsSummary:
    """
    Analyze a single column from the calibration DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    column : str
        Column name to analyze.

    Returns
    -------
    StatsSummary
        Statistics summary for the column.
    """
    data = df[column].dropna().values
    return compute_stats_summary(data, name=column)


def analyze_all_numeric_columns(df: pd.DataFrame) -> Dict[str, StatsSummary]:
    """
    Analyze all numeric columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    dict
        Dictionary mapping column names to StatsSummary objects.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['Qubit']  # Don't analyze qubit numbers

    results = {}
    for col in numeric_cols:
        if col not in exclude_cols:
            results[col] = analyze_column(df, col)

    return results


def get_outlier_qubits(
    df: pd.DataFrame, column: str, iqr_factor: float = 1.5
) -> Tuple[List[int], List[int]]:
    """
    Get qubit numbers that are outliers for a given metric.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    column : str
        Column to analyze.
    iqr_factor : float
        IQR factor for outlier detection.

    Returns
    -------
    tuple
        (low_outlier_qubits, high_outlier_qubits)
    """
    data = df[column].values
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    low_threshold = q1 - iqr_factor * iqr
    high_threshold = q3 + iqr_factor * iqr

    low_outliers = df[df[column] < low_threshold]['Qubit'].tolist()
    high_outliers = df[df[column] > high_threshold]['Qubit'].tolist()

    return low_outliers, high_outliers


def get_cz_outlier_edges(
    df: pd.DataFrame, iqr_factor: float = 1.5
) -> pd.DataFrame:
    """
    Get CZ edges that are outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    iqr_factor : float
        IQR factor for outlier detection.

    Returns
    -------
    pd.DataFrame
        DataFrame of outlier edges with their errors.
    """
    edge_df = create_edge_dataframe(df)
    errors = edge_df['cz_error'].values

    mask, _, low, high = detect_outliers_iqr(errors, factor=iqr_factor)
    return edge_df[mask].sort_values('cz_error', ascending=False)


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    numeric_cols = [
        'T1 (us)', 'T2 (us)', 'Readout assignment error',
        'Prob meas0 prep1', 'Prob meas1 prep0', 'ID error',
        'RX error', 'MEASURE error'
    ]
    existing_cols = [c for c in numeric_cols if c in df.columns]
    return df[existing_cols].corr()


def t1_t2_ratio_analysis(df: pd.DataFrame) -> Dict[str, Union[float, List[int]]]:
    """
    Analyze T1/T2 ratio (T2 should ideally be <= 2*T1).

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    dict
        Analysis results including qubits violating the T2 <= 2*T1 constraint.
    """
    t1 = df['T1 (us)'].values
    t2 = df['T2 (us)'].values
    ratio = t2 / t1

    # T2 should be <= 2*T1 (fundamental physics limit)
    violating_mask = t2 > 2 * t1
    violating_qubits = df[violating_mask]['Qubit'].tolist()

    return {
        'mean_ratio': np.mean(ratio),
        'median_ratio': np.median(ratio),
        'min_ratio': np.min(ratio),
        'max_ratio': np.max(ratio),
        'violating_qubits': violating_qubits,
        'violation_count': len(violating_qubits),
    }


def readout_error_analysis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze readout errors and asymmetry.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    dict
        Readout error analysis results.
    """
    p0_1 = df['Prob meas0 prep1'].values  # False negative
    p1_0 = df['Prob meas1 prep0'].values  # False positive

    asymmetry = np.abs(p0_1 - p1_0)

    return {
        'mean_p0_1': np.mean(p0_1),
        'mean_p1_0': np.mean(p1_0),
        'mean_assignment_error': df['Readout assignment error'].mean(),
        'mean_asymmetry': np.mean(asymmetry),
        'max_asymmetry': np.max(asymmetry),
        'max_asymmetry_qubit': df.loc[np.argmax(asymmetry), 'Qubit'],
    }


def connection_statistics(df: pd.DataFrame) -> Dict[str, Union[int, float, Dict]]:
    """
    Analyze qubit connectivity statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    dict
        Connectivity statistics.
    """
    connections = count_qubit_connections(df)
    conn_values = list(connections.values())

    # Group qubits by connection count
    conn_groups = {}
    for qubit, count in connections.items():
        if count not in conn_groups:
            conn_groups[count] = []
        conn_groups[count].append(qubit)

    return {
        'total_qubits': len(connections),
        'mean_connections': np.mean(conn_values),
        'connection_distribution': {k: len(v) for k, v in conn_groups.items()},
        'qubits_by_connections': conn_groups,
    }


def generate_summary_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive text summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    str
        Formatted summary report.
    """
    report = []
    report.append("=" * 70)
    report.append("IBM MIAMI QUANTUM DEVICE CALIBRATION SUMMARY")
    report.append("=" * 70)

    # Basic info
    report.append(f"\nTotal Qubits: {len(df)}")
    operational = df[df['Operational'] == 'Yes']
    report.append(f"Operational Qubits: {len(operational)}")

    # Connectivity
    conn_stats = connection_statistics(df)
    report.append(f"\n--- Connectivity ---")
    report.append(f"Connection Distribution: {conn_stats['connection_distribution']}")

    # T1/T2
    report.append(f"\n--- Coherence Times ---")
    t1_stats = compute_stats_summary(df['T1 (us)'].values, "T1 (us)")
    t2_stats = compute_stats_summary(df['T2 (us)'].values, "T2 (us)")
    report.append(f"T1: mean={t1_stats.mean:.1f}us, median={t1_stats.median:.1f}us, "
                  f"range=[{t1_stats.min_val:.1f}, {t1_stats.max_val:.1f}]us")
    report.append(f"T2: mean={t2_stats.mean:.1f}us, median={t2_stats.median:.1f}us, "
                  f"range=[{t2_stats.min_val:.1f}, {t2_stats.max_val:.1f}]us")
    report.append(f"T1 outliers (low): {t1_stats.outlier_count}")
    report.append(f"T2 outliers: {t2_stats.outlier_count}")

    # CZ Errors
    report.append(f"\n--- Two-Qubit (CZ) Gate Errors ---")
    cz_errors = get_cz_errors_array(df)
    cz_stats = compute_stats_summary(cz_errors, "CZ Error")
    report.append(f"Total CZ edges: {len(cz_errors)}")
    report.append(f"Mean: {cz_stats.mean:.6f}, Median: {cz_stats.median:.6f}")
    report.append(f"Range: [{cz_stats.min_val:.6f}, {cz_stats.max_val:.6f}]")
    report.append(f"Outliers (high error): {cz_stats.outlier_count}")

    # Gate lengths
    report.append(f"\n--- CZ Gate Lengths ---")
    gate_lengths = get_gate_lengths_array(df)
    gl_stats = compute_stats_summary(gate_lengths.astype(float), "Gate Length (ns)")
    report.append(f"Mean: {gl_stats.mean:.1f}ns, Median: {gl_stats.median:.1f}ns")
    report.append(f"Range: [{gl_stats.min_val:.0f}, {gl_stats.max_val:.0f}]ns")

    # Readout
    report.append(f"\n--- Readout Errors ---")
    ro_analysis = readout_error_analysis(df)
    report.append(f"Mean assignment error: {ro_analysis['mean_assignment_error']:.6f}")
    report.append(f"Mean P(0|1) [false neg]: {ro_analysis['mean_p0_1']:.6f}")
    report.append(f"Mean P(1|0) [false pos]: {ro_analysis['mean_p1_0']:.6f}")

    # Single-qubit errors
    report.append(f"\n--- Single-Qubit Gate Errors ---")
    id_stats = compute_stats_summary(df['ID error'].values, "ID Error")
    report.append(f"ID Error: mean={id_stats.mean:.6f}, median={id_stats.median:.6f}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)


if __name__ == "__main__":
    import glob

    csv_files = glob.glob("*.csv")
    if csv_files:
        df = load_calibration_data(csv_files[0])
        print(generate_summary_report(df))

        print("\n\n--- Detailed Column Analysis ---\n")
        all_stats = analyze_all_numeric_columns(df)
        for col, stats_summary in all_stats.items():
            print(stats_summary)
