"""
IBM Miami Quantum Device Calibration Data Loader

This module handles loading and parsing of IBM Miami calibration CSV data,
including special handling for multi-value columns like CZ error and Gate length.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def load_calibration_data(csv_path: str) -> pd.DataFrame:
    """
    Load IBM Miami calibration data from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the calibration CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with calibration data.
    """
    df = pd.read_csv(csv_path)
    df['Qubit'] = df['Qubit'].astype(int)
    return df


def parse_cz_errors(cz_error_str: str) -> Dict[int, float]:
    """
    Parse CZ error string into dictionary of neighbor qubit -> error.

    Format: "neighbor1:error1;neighbor2:error2;..."
    Example: "1:0.00128;10:0.00107" -> {1: 0.00128, 10: 0.00107}

    Parameters
    ----------
    cz_error_str : str
        CZ error string from the CSV.

    Returns
    -------
    dict
        Dictionary mapping neighbor qubit number to error value.
    """
    if pd.isna(cz_error_str) or cz_error_str == '':
        return {}

    result = {}
    pairs = cz_error_str.split(';')
    for pair in pairs:
        if ':' in pair:
            neighbor, error = pair.split(':')
            result[int(neighbor)] = float(error)
    return result


def parse_gate_lengths(gate_length_str: str) -> Dict[int, int]:
    """
    Parse gate length string into dictionary of neighbor qubit -> length.

    Format: "neighbor1:length1;neighbor2:length2;..."
    Example: "1:128;10:128" -> {1: 128, 10: 128}

    Parameters
    ----------
    gate_length_str : str
        Gate length string from the CSV.

    Returns
    -------
    dict
        Dictionary mapping neighbor qubit number to gate length in ns.
    """
    if pd.isna(gate_length_str) or gate_length_str == '':
        return {}

    result = {}
    pairs = gate_length_str.split(';')
    for pair in pairs:
        if ':' in pair:
            neighbor, length = pair.split(':')
            result[int(neighbor)] = int(length)
    return result


def extract_all_cz_errors(df: pd.DataFrame) -> List[Tuple[int, int, float]]:
    """
    Extract all CZ errors as a list of (qubit1, qubit2, error) tuples.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    list
        List of (qubit1, qubit2, error) tuples for all CZ gates.
    """
    all_errors = []
    for _, row in df.iterrows():
        qubit = row['Qubit']
        cz_dict = parse_cz_errors(row['CZ error'])
        for neighbor, error in cz_dict.items():
            # Only add each pair once (when qubit < neighbor)
            if qubit < neighbor:
                all_errors.append((qubit, neighbor, error))
    return all_errors


def extract_all_gate_lengths(df: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """
    Extract all CZ gate lengths as a list of (qubit1, qubit2, length) tuples.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    list
        List of (qubit1, qubit2, length) tuples for all CZ gates.
    """
    all_lengths = []
    for _, row in df.iterrows():
        qubit = row['Qubit']
        length_dict = parse_gate_lengths(row['Gate length (ns)'])
        for neighbor, length in length_dict.items():
            # Only add each pair once (when qubit < neighbor)
            if qubit < neighbor:
                all_lengths.append((qubit, neighbor, length))
    return all_lengths


def get_cz_errors_array(df: pd.DataFrame) -> np.ndarray:
    """
    Get array of all CZ error values.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    np.ndarray
        Array of all CZ error values.
    """
    errors = extract_all_cz_errors(df)
    return np.array([e[2] for e in errors])


def get_gate_lengths_array(df: pd.DataFrame) -> np.ndarray:
    """
    Get array of all CZ gate length values.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    np.ndarray
        Array of all CZ gate length values in ns.
    """
    lengths = extract_all_gate_lengths(df)
    return np.array([l[2] for l in lengths])


def count_qubit_connections(df: pd.DataFrame) -> Dict[int, int]:
    """
    Count number of CZ connections for each qubit.

    For a planar device, boundary qubits have fewer than 4 connections.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    dict
        Dictionary mapping qubit number to connection count.
    """
    connections = {}
    for _, row in df.iterrows():
        qubit = row['Qubit']
        cz_dict = parse_cz_errors(row['CZ error'])
        connections[qubit] = len(cz_dict)
    return connections


def identify_boundary_qubits(df: pd.DataFrame, max_connections: int = 4) -> List[int]:
    """
    Identify boundary qubits (those with fewer than max_connections).

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    max_connections : int
        Maximum expected connections for interior qubits (default 4).

    Returns
    -------
    list
        List of boundary qubit numbers.
    """
    connections = count_qubit_connections(df)
    return [q for q, c in connections.items() if c < max_connections]


def identify_interior_qubits(df: pd.DataFrame, max_connections: int = 4) -> List[int]:
    """
    Identify interior qubits (those with max_connections).

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    max_connections : int
        Expected connections for interior qubits (default 4).

    Returns
    -------
    list
        List of interior qubit numbers.
    """
    connections = count_qubit_connections(df)
    return [q for q, c in connections.items() if c == max_connections]


def create_edge_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame with one row per CZ edge.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: qubit1, qubit2, cz_error, gate_length_ns
    """
    cz_errors = extract_all_cz_errors(df)
    gate_lengths = extract_all_gate_lengths(df)

    # Create dictionaries for lookup
    length_dict = {(q1, q2): length for q1, q2, length in gate_lengths}

    edges = []
    for q1, q2, error in cz_errors:
        length = length_dict.get((q1, q2), np.nan)
        edges.append({
            'qubit1': q1,
            'qubit2': q2,
            'cz_error': error,
            'gate_length_ns': length
        })

    return pd.DataFrame(edges)


if __name__ == "__main__":
    # Example usage
    import glob

    csv_files = glob.glob("*.csv")
    if csv_files:
        df = load_calibration_data(csv_files[0])
        print(f"Loaded {len(df)} qubits")
        print(f"\nColumns: {list(df.columns)}")

        # Show boundary vs interior qubits
        boundary = identify_boundary_qubits(df)
        interior = identify_interior_qubits(df)
        print(f"\nBoundary qubits ({len(boundary)}): {boundary[:10]}...")
        print(f"Interior qubits ({len(interior)}): {interior[:10]}...")

        # Show CZ error stats
        cz_errors = get_cz_errors_array(df)
        print(f"\nCZ Errors: {len(cz_errors)} edges")
        print(f"  Mean: {cz_errors.mean():.6f}")
        print(f"  Median: {np.median(cz_errors):.6f}")
        print(f"  Min: {cz_errors.min():.6f}")
        print(f"  Max: {cz_errors.max():.6f}")
