"""
IBM Miami Quantum Device Calibration Report Generator

Generates a comprehensive PDF report suitable for physicists,
with explanations of all calibration parameters and their physical significance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from ibm_miami_loader import (
    load_calibration_data,
    get_cz_errors_array,
    get_gate_lengths_array,
    extract_all_cz_errors,
    count_qubit_connections,
    identify_boundary_qubits,
    identify_interior_qubits,
    create_edge_dataframe,
    parse_cz_errors,
)

from ibm_miami_stats import (
    compute_stats_summary,
    detect_outliers_iqr,
    t1_t2_ratio_analysis,
    readout_error_analysis,
    connection_statistics,
)


# ===========================================================================
# Physical Explanations for Report
# ===========================================================================

DEVICE_DESCRIPTION = """
IBM Miami is a 120-qubit superconducting quantum processor based on IBM's heavy-hex lattice
topology. Each qubit is a transmon - an anharmonic oscillator formed by a Josephson junction
shunted by a capacitor. The device operates at millikelvin temperatures (~15 mK) in a dilution
refrigerator to minimize thermal excitations.

The heavy-hex topology arranges qubits in a pattern where most interior qubits have 4 nearest
neighbors, while boundary qubits have 2-3 neighbors. This geometry is optimized for error
correction codes while maintaining high-fidelity two-qubit gates.
"""

COLUMN_EXPLANATIONS = {
    'Qubit': """
**Qubit Index**
Integer identifier (0-119) for each physical qubit on the chip. The numbering follows
the chip's spatial layout in the heavy-hex lattice topology.
""",

    'T1 (us)': """
**T1 Relaxation Time (Energy Decay)**
T1 characterizes the timescale for energy relaxation from the excited state |1⟩ to the
ground state |0⟩. This is analogous to the longitudinal relaxation time in NMR.

Physical mechanism: The qubit couples to its electromagnetic environment (substrate
defects, quasiparticles, radiation), causing spontaneous emission. The decay follows:
    P₁(t) = P₁(0) × exp(-t/T1)

Longer T1 allows more gate operations before decoherence. Values >300 μs are considered
excellent for superconducting qubits. Low T1 often indicates material defects or
spurious coupling to lossy modes.
""",

    'T2 (us)': """
**T2 Dephasing Time (Phase Coherence)**
T2 characterizes the timescale for loss of phase coherence in a superposition state.
For a state |ψ⟩ = (|0⟩ + |1⟩)/√2, the off-diagonal density matrix elements decay as exp(-t/T2).

Physical mechanisms:
- T1 processes (energy decay also causes dephasing)
- Pure dephasing from low-frequency noise (flux noise, charge noise)

Fundamental limit: T2 ≤ 2×T1 (from Bloch equations). T2 approaching 2×T1 indicates
T1-limited dephasing with minimal pure dephasing - a sign of a well-isolated qubit.

T2 is typically measured via Ramsey or Hahn echo sequences. Values reported here are
likely T2* (Ramsey) or T2 echo, depending on IBM's calibration protocol.
""",

    'Readout assignment error': """
**Readout Assignment Error**
The probability of incorrectly assigning the qubit state after measurement. This is the
average of the two types of misclassification:
    ε_readout = (P(0|1) + P(1|0)) / 2

Modern superconducting qubit readout uses dispersive measurement: the qubit state shifts
the resonant frequency of a coupled readout resonator. A microwave pulse probes this
resonator, and the reflected/transmitted signal is amplified and digitized.

Errors arise from:
- Finite signal-to-noise ratio in the measurement chain
- T1 decay during measurement (state flip before readout completes)
- Readout resonator thermal population
- Imperfect state discrimination thresholds
""",

    'Prob meas0 prep1': """
**P(0|1) - False Negative Rate**
Probability of measuring |0⟩ when the qubit was prepared in |1⟩. This is primarily
caused by T1 decay during the measurement process.

If the readout duration is τ_read and T1 is the relaxation time:
    P(0|1) ≈ 1 - exp(-τ_read / T1)

High P(0|1) often correlates with low T1 or long readout pulses.
""",

    'Prob meas1 prep0': """
**P(1|0) - False Positive Rate**
Probability of measuring |1⟩ when the qubit was prepared in |0⟩. This can arise from:
- Thermal excitation of the qubit
- Measurement-induced excitation (QND violation)
- Readout resonator thermal population creating spurious signals
- Discrimination threshold errors

This error is typically smaller than P(0|1) since there's no T1 decay channel
from |0⟩ to |1⟩ at millikelvin temperatures.
""",

    'Readout length (ns)': """
**Readout Pulse Duration**
The duration of the microwave pulse used to probe the readout resonator. Longer pulses
provide better signal-to-noise (more photons) but increase the window for T1 decay.

Typical values: 500-3000 ns. The optimal length balances SNR against T1 decay.
IBM Miami uses 2400 ns uniformly across all qubits.
""",

    'ID error': """
**Identity Gate Error**
Error rate of the identity (idle) operation over one gate cycle. This benchmarks
decoherence during the time a qubit waits while other qubits are being operated on.

Measured via randomized benchmarking (RB) or similar protocols. The identity gate
error sets a floor for all single-qubit operations since any gate includes at least
this much decoherence.

ID error ≈ (gate_time / T1 + gate_time / T2) / 2 for ideal gates.
""",

    'Single-qubit gate length (ns)': """
**Single-Qubit Gate Duration**
Time to execute a single-qubit gate (e.g., X, Y, √X). For transmon qubits, these are
microwave pulses at the qubit transition frequency.

Typical values: 20-50 ns. Shorter gates reduce decoherence during operations but
require higher drive power (risking leakage to non-computational states).

IBM Miami uses 32 ns gates, which is relatively fast.
""",

    'RX error': """
**RX Gate Error (X-rotation)**
Error rate for arbitrary rotations around the X-axis of the Bloch sphere. Measured
via randomized benchmarking.

Physical implementation: A resonant microwave pulse with controlled amplitude and
duration. Errors arise from:
- Pulse calibration imperfections (amplitude, frequency, phase)
- Decoherence during the gate
- Leakage to higher transmon levels
- Crosstalk from neighboring qubits
""",

    'Z-axis rotation (rz) error': """
**RZ Gate Error (Z-rotation)**
Error rate for rotations around the Z-axis. In IBM's implementation, RZ gates are
"virtual" - implemented by adjusting the phase of subsequent pulses rather than
applying a physical pulse.

Virtual RZ gates have essentially zero error (only software phase tracking), which
is why this column shows 0 for all qubits. This is a standard technique in
superconducting qubit control.
""",

    '√x (sx) error': """
**SX Gate Error (√X gate)**
Error rate for the square-root of X gate, which performs a π/2 rotation around X.
This is one of IBM's native gate set elements (along with RZ and CZ/ECR).

SX followed by SX equals X. Using SX as a native gate allows efficient decomposition
of arbitrary single-qubit rotations.
""",

    'Pauli-X error': """
**X Gate Error (Bit Flip)**
Error rate for the Pauli-X gate (π rotation around X), equivalent to a classical
NOT operation: |0⟩ ↔ |1⟩.

Typically X = SX × SX, so X error ≈ 2 × SX error (approximately, neglecting correlations).
""",

    'CZ error': """
**CZ (Controlled-Z) Gate Error**
Error rate for the two-qubit controlled-Z gate between connected qubit pairs.
CZ applies a π phase to the |11⟩ state: CZ|11⟩ = -|11⟩.

Format: "neighbor1:error1;neighbor2:error2;..." listing all connected qubits and
their respective CZ error rates.

Physical implementation varies (cross-resonance, tunable coupler, etc.). Two-qubit
gates are typically 10-100× noisier than single-qubit gates due to:
- Longer gate duration
- Coupling to additional decoherence channels
- Calibration sensitivity
- Crosstalk and spectator qubit errors

CZ errors are the dominant error source in most quantum algorithms.
""",

    'Gate length (ns)': """
**CZ Gate Duration**
Duration of the two-qubit CZ gate for each qubit pair.

Format mirrors CZ error: "neighbor1:duration1;neighbor2:duration2;..."

Typical values: 100-500 ns. Duration varies by qubit pair due to:
- Frequency detuning between qubits
- Coupling strength
- Required pulse shaping for leakage suppression

Longer gates generally have higher error rates due to increased decoherence.
""",

    'MEASURE error': """
**Measurement Error**
Error rate for the measurement operation, typically equal to the readout assignment
error. May include additional systematic effects from the measurement protocol.

Note: MEASURE error equals Readout assignment error in this dataset.
""",

    'Operational': """
**Operational Status**
Whether the qubit passed calibration and is available for use. "Yes" indicates the
qubit meets IBM's quality thresholds. Failed qubits would show "No" and should be
avoided in circuit design.

All 120 qubits are operational in this calibration snapshot.
"""
}


def create_text_page(pdf: PdfPages, title: str, text: str, fontsize: int = 10):
    """Create a page with formatted text."""
    fig = plt.figure(figsize=(8.5, 11))

    # Title
    fig.text(0.5, 0.95, title, fontsize=14, fontweight='bold',
             ha='center', va='top')

    # Body text
    fig.text(0.1, 0.88, text, fontsize=fontsize,
             ha='left', va='top', wrap=True,
             transform=fig.transFigure,
             fontfamily='monospace' if '|' in text else 'serif')

    # Add margins
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.axis('off')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_title_page(pdf: PdfPages, csv_path: str):
    """Create the report title page."""
    fig = plt.figure(figsize=(8.5, 11))

    fig.text(0.5, 0.7, "IBM Miami Quantum Processor", fontsize=24,
             fontweight='bold', ha='center')
    fig.text(0.5, 0.62, "Calibration Data Analysis Report", fontsize=18,
             ha='center')

    fig.text(0.5, 0.50, "120-Qubit Superconducting Quantum Computer",
             fontsize=12, ha='center', style='italic')

    fig.text(0.5, 0.35, f"Data Source: {csv_path}", fontsize=10, ha='center')
    fig.text(0.5, 0.30, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             fontsize=10, ha='center')

    fig.text(0.5, 0.15, "Prepared for: Quantum Physics Research",
             fontsize=10, ha='center')

    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)


def create_toc_page(pdf: PdfPages):
    """Create table of contents page."""
    toc = """
TABLE OF CONTENTS

1. Executive Summary ................................. 3
2. Device Overview ................................... 4
3. Parameter Definitions ............................. 5-10
4. Coherence Analysis (T1, T2) ....................... 11-12
5. Two-Qubit Gate Analysis (CZ) ...................... 13-14
6. Readout Error Analysis ............................ 15-16
7. Single-Qubit Gate Analysis ........................ 17-18
8. Device Connectivity ............................... 19
9. Correlation Analysis .............................. 20
10. Outlier Identification ........................... 21
11. Best and Worst Qubits ............................ 22
12. Statistical Summary Tables ....................... 23-24
"""
    create_text_page(pdf, "Table of Contents", toc)


def create_executive_summary(pdf: PdfPages, df: pd.DataFrame):
    """Create executive summary page."""
    cz_errors = get_cz_errors_array(df)
    gate_lengths = get_gate_lengths_array(df)
    conn_stats = connection_statistics(df)

    t1_mean = df['T1 (us)'].mean()
    t2_mean = df['T2 (us)'].mean()

    summary = f"""
EXECUTIVE SUMMARY

Device: IBM Miami (ibm_miami)
Topology: Heavy-hex lattice
Total Qubits: {len(df)}
Operational Qubits: {len(df[df['Operational'] == 'Yes'])}
Total CZ Edges: {len(cz_errors)}

CONNECTIVITY
  Interior qubits (4 neighbors): {conn_stats['connection_distribution'].get(4, 0)}
  Edge qubits (3 neighbors): {conn_stats['connection_distribution'].get(3, 0)}
  Corner qubits (2 neighbors): {conn_stats['connection_distribution'].get(2, 0)}

COHERENCE TIMES
  T1: {t1_mean:.1f} +/- {df['T1 (us)'].std():.1f} us (range: {df['T1 (us)'].min():.1f} - {df['T1 (us)'].max():.1f} us)
  T2: {t2_mean:.1f} +/- {df['T2 (us)'].std():.1f} us (range: {df['T2 (us)'].min():.1f} - {df['T2 (us)'].max():.1f} us)
  T2/T1 ratio: {(t2_mean/t1_mean):.2f} (ideal limit: 2.0)

GATE ERRORS
  Single-qubit (ID): {df['ID error'].mean()*100:.3f}% mean ({df['ID error'].median()*100:.3f}% median)
  Two-qubit (CZ): {cz_errors.mean()*100:.2f}% mean ({np.median(cz_errors)*100:.2f}% median)
  CZ gate length: {gate_lengths.mean():.0f} +/- {gate_lengths.std():.0f} ns

READOUT
  Assignment error: {df['Readout assignment error'].mean()*100:.2f}% mean
  P(0|1) [false negative]: {df['Prob meas0 prep1'].mean()*100:.2f}%
  P(1|0) [false positive]: {df['Prob meas1 prep0'].mean()*100:.2f}%

KEY OBSERVATIONS
  - CZ error distribution is highly right-skewed (mean >> median)
  - {len(detect_outliers_iqr(cz_errors)[1])} CZ edges are outliers (>{np.percentile(cz_errors, 75) + 1.5*(np.percentile(cz_errors, 75)-np.percentile(cz_errors, 25)):.3f})
  - Worst CZ error: {cz_errors.max()*100:.1f}% (qubits 6-7)
  - 3 qubits violate T2 <= 2*T1 physical limit (measurement artifact or pulse errors)
"""
    create_text_page(pdf, "Executive Summary", summary, fontsize=9)


def create_parameter_pages(pdf: PdfPages):
    """Create pages explaining each parameter."""
    # Group parameters for multi-column pages
    params_page1 = ['Qubit', 'T1 (us)', 'T2 (us)']
    params_page2 = ['Readout assignment error', 'Prob meas0 prep1', 'Prob meas1 prep0']
    params_page3 = ['Readout length (ns)', 'ID error', 'Single-qubit gate length (ns)']
    params_page4 = ['RX error', 'Z-axis rotation (rz) error', '√x (sx) error', 'Pauli-X error']
    params_page5 = ['CZ error', 'Gate length (ns)']
    params_page6 = ['MEASURE error', 'Operational']

    for i, params in enumerate([params_page1, params_page2, params_page3,
                                 params_page4, params_page5, params_page6]):
        text = ""
        for p in params:
            if p in COLUMN_EXPLANATIONS:
                text += COLUMN_EXPLANATIONS[p] + "\n"
        create_text_page(pdf, f"Parameter Definitions ({i+1}/6)", text.strip(), fontsize=9)


def plot_coherence_analysis(pdf: PdfPages, df: pd.DataFrame):
    """Create coherence time analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Coherence Time Analysis (T1, T2)', fontsize=14, fontweight='bold')

    t1 = df['T1 (us)'].values
    t2 = df['T2 (us)'].values

    # T1 histogram
    ax = axes[0, 0]
    ax.hist(t1, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(t1), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(t1):.0f} μs')
    ax.axvline(np.median(t1), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(t1):.0f} μs')
    ax.set_xlabel('T1 (μs)')
    ax.set_ylabel('Count')
    ax.set_title('T1 Relaxation Time Distribution')
    ax.legend()

    # T2 histogram
    ax = axes[0, 1]
    ax.hist(t2, bins=30, edgecolor='black', alpha=0.7, color='darkorange')
    ax.axvline(np.mean(t2), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(t2):.0f} μs')
    ax.axvline(np.median(t2), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(t2):.0f} μs')
    ax.set_xlabel('T2 (μs)')
    ax.set_ylabel('Count')
    ax.set_title('T2 Dephasing Time Distribution')
    ax.legend()

    # T1 vs T2 scatter
    ax = axes[1, 0]
    scatter = ax.scatter(t1, t2, c=df['Qubit'], cmap='viridis', alpha=0.7, s=30)
    max_val = max(t1.max(), t2.max())
    ax.plot([0, max_val], [0, 2*max_val], 'r--', alpha=0.7, linewidth=2, label='T2 = 2×T1 (physical limit)')
    ax.plot([0, max_val], [0, max_val], 'k:', alpha=0.5, label='T2 = T1')
    ax.set_xlabel('T1 (μs)')
    ax.set_ylabel('T2 (μs)')
    ax.set_title('T1 vs T2 Correlation')
    ax.legend(loc='upper left')
    plt.colorbar(scatter, ax=ax, label='Qubit Index')

    # T2/T1 ratio histogram
    ax = axes[1, 1]
    ratio = t2 / t1
    ax.hist(ratio, bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Physical limit (T2 = 2×T1)')
    ax.axvline(np.median(ratio), color='green', linestyle='-', linewidth=2, label=f'Median: {np.median(ratio):.2f}')
    ax.set_xlabel('T2/T1 Ratio')
    ax.set_ylabel('Count')
    ax.set_title('T2/T1 Ratio Distribution')
    ax.legend()

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Second page: T1 and T2 by qubit
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle('Coherence Times by Qubit', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.bar(df['Qubit'], t1, color='steelblue', alpha=0.7)
    ax.axhline(np.mean(t1), color='red', linestyle='--', label=f'Mean: {np.mean(t1):.0f} μs')
    ax.axhline(100, color='orange', linestyle=':', alpha=0.7, label='100 μs threshold')
    ax.set_xlabel('Qubit Index')
    ax.set_ylabel('T1 (μs)')
    ax.set_title('T1 by Qubit (sorted by qubit index)')
    ax.legend()
    ax.set_xlim(-1, 120)

    ax = axes[1]
    ax.bar(df['Qubit'], t2, color='darkorange', alpha=0.7)
    ax.axhline(np.mean(t2), color='red', linestyle='--', label=f'Mean: {np.mean(t2):.0f} μs')
    ax.axhline(100, color='blue', linestyle=':', alpha=0.7, label='100 μs threshold')
    ax.set_xlabel('Qubit Index')
    ax.set_ylabel('T2 (μs)')
    ax.set_title('T2 by Qubit (sorted by qubit index)')
    ax.legend()
    ax.set_xlim(-1, 120)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_cz_analysis(pdf: PdfPages, df: pd.DataFrame):
    """Create CZ gate error analysis plots."""
    cz_errors = get_cz_errors_array(df)
    gate_lengths = get_gate_lengths_array(df)
    edge_df = create_edge_dataframe(df)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Two-Qubit (CZ) Gate Analysis', fontsize=14, fontweight='bold')

    # CZ error histogram (linear)
    ax = axes[0, 0]
    ax.hist(cz_errors, bins=50, edgecolor='black', alpha=0.7, color='crimson')
    ax.axvline(np.mean(cz_errors), color='blue', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(cz_errors)*100:.2f}%')
    ax.axvline(np.median(cz_errors), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(cz_errors)*100:.2f}%')
    ax.set_xlabel('CZ Error')
    ax.set_ylabel('Count')
    ax.set_title('CZ Error Distribution (Linear Scale)')
    ax.legend()

    # CZ error histogram (log scale)
    ax = axes[0, 1]
    log_bins = np.logspace(np.log10(cz_errors.min()), np.log10(cz_errors.max()), 40)
    ax.hist(cz_errors, bins=log_bins, edgecolor='black', alpha=0.7, color='crimson')
    ax.set_xscale('log')
    ax.axvline(np.median(cz_errors), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(cz_errors)*100:.2f}%')
    # IQR outlier threshold
    q1, q3 = np.percentile(cz_errors, [25, 75])
    threshold = q3 + 1.5 * (q3 - q1)
    ax.axvline(threshold, color='red', linestyle=':', linewidth=2,
               label=f'Outlier threshold: {threshold*100:.2f}%')
    ax.set_xlabel('CZ Error (log scale)')
    ax.set_ylabel('Count')
    ax.set_title('CZ Error Distribution (Log Scale) - Note Right Skew')
    ax.legend()

    # Gate length histogram
    ax = axes[1, 0]
    ax.hist(gate_lengths, bins=30, edgecolor='black', alpha=0.7, color='teal')
    ax.axvline(np.mean(gate_lengths), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(gate_lengths):.0f} ns')
    ax.set_xlabel('Gate Length (ns)')
    ax.set_ylabel('Count')
    ax.set_title('CZ Gate Duration Distribution')
    ax.legend()

    # CZ error vs gate length
    ax = axes[1, 1]
    sc = ax.scatter(edge_df['gate_length_ns'], edge_df['cz_error']*100,
                    alpha=0.5, c='crimson', s=20)
    ax.set_xlabel('Gate Length (ns)')
    ax.set_ylabel('CZ Error (%)')
    ax.set_title('CZ Error vs Gate Duration')
    ax.set_yscale('log')

    # Add correlation annotation
    corr = edge_df['gate_length_ns'].corr(edge_df['cz_error'])
    ax.text(0.95, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Second page: worst CZ edges
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.suptitle('Worst CZ Gate Errors (Top 30)', fontsize=14, fontweight='bold')

    worst = edge_df.nlargest(30, 'cz_error')
    labels = [f'{int(r.qubit1)}-{int(r.qubit2)}' for _, r in worst.iterrows()]
    errors = worst['cz_error'].values * 100

    bars = ax.barh(range(len(labels)), errors, color='crimson', alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('CZ Error (%)')
    ax.set_ylabel('Qubit Pair')
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, err) in enumerate(zip(bars, errors)):
        ax.text(err + 0.2, i, f'{err:.2f}%', va='center', fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_readout_analysis(pdf: PdfPages, df: pd.DataFrame):
    """Create readout error analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Readout/Measurement Error Analysis', fontsize=14, fontweight='bold')

    ro_error = df['Readout assignment error'].values
    p01 = df['Prob meas0 prep1'].values
    p10 = df['Prob meas1 prep0'].values

    # Readout error histogram
    ax = axes[0, 0]
    ax.hist(ro_error*100, bins=40, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.axvline(np.mean(ro_error)*100, color='red', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(ro_error)*100:.2f}%')
    ax.axvline(np.median(ro_error)*100, color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(ro_error)*100:.2f}%')
    ax.set_xlabel('Readout Assignment Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('Readout Error Distribution')
    ax.legend()

    # P(0|1) vs P(1|0) scatter - asymmetry analysis
    ax = axes[0, 1]
    ax.scatter(p01*100, p10*100, c=df['Qubit'], cmap='viridis', alpha=0.7, s=30)
    max_val = max(p01.max(), p10.max()) * 100
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Equal errors')
    ax.set_xlabel('P(0|1) - Prepared |1⟩, Measured |0⟩ (%)')
    ax.set_ylabel('P(1|0) - Prepared |0⟩, Measured |1⟩ (%)')
    ax.set_title('Readout Error Asymmetry (P(0|1) typically > P(1|0) due to T1)')
    ax.legend()

    # Readout error by qubit (log scale for outliers)
    ax = axes[1, 0]
    ax.bar(df['Qubit'], ro_error*100, color='mediumpurple', alpha=0.7)
    ax.axhline(np.mean(ro_error)*100, color='red', linestyle='--',
               label=f'Mean: {np.mean(ro_error)*100:.2f}%')
    ax.set_xlabel('Qubit Index')
    ax.set_ylabel('Readout Error (%)')
    ax.set_title('Readout Error by Qubit')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(-1, 120)

    # P(0|1) and P(1|0) comparison
    ax = axes[1, 1]
    width = 0.35
    x = np.arange(len(df))
    # Only show every 5th qubit for readability
    mask = df['Qubit'] % 5 == 0
    x_show = df[mask]['Qubit'].values
    p01_show = df[mask]['Prob meas0 prep1'].values * 100
    p10_show = df[mask]['Prob meas1 prep0'].values * 100

    ax.bar(x_show - width/2, p01_show, width, label='P(0|1) - T1 decay', color='coral', alpha=0.7)
    ax.bar(x_show + width/2, p10_show, width, label='P(1|0) - Thermal/other', color='skyblue', alpha=0.7)
    ax.set_xlabel('Qubit Index (every 5th)')
    ax.set_ylabel('Error Probability (%)')
    ax.set_title('False Negative vs False Positive by Qubit')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Second page: worst readout qubits
    fig, ax = plt.subplots(figsize=(11, 8.5))

    worst = df.nlargest(20, 'Readout assignment error')[
        ['Qubit', 'Readout assignment error', 'Prob meas0 prep1', 'Prob meas1 prep0', 'T1 (us)']
    ]

    ax.axis('off')
    table_data = [['Qubit', 'Assignment Error', 'P(0|1)', 'P(1|0)', 'T1 (μs)']]
    for _, row in worst.iterrows():
        table_data.append([
            f"{int(row['Qubit'])}",
            f"{row['Readout assignment error']*100:.2f}%",
            f"{row['Prob meas0 prep1']*100:.2f}%",
            f"{row['Prob meas1 prep0']*100:.2f}%",
            f"{row['T1 (us)']:.1f}"
        ])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.12, 0.22, 0.18, 0.18, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Header styling
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Top 20 Worst Readout Qubits\n(Note correlation between high P(0|1) and T1)',
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_single_qubit_analysis(pdf: PdfPages, df: pd.DataFrame):
    """Create single-qubit gate error analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Single-Qubit Gate Error Analysis', fontsize=14, fontweight='bold')

    id_error = df['ID error'].values
    rx_error = df['RX error'].values

    # ID error histogram (log scale)
    ax = axes[0, 0]
    log_bins = np.logspace(np.log10(id_error.min()), np.log10(id_error.max()), 40)
    ax.hist(id_error, bins=log_bins, edgecolor='black', alpha=0.7, color='forestgreen')
    ax.set_xscale('log')
    ax.axvline(np.median(id_error), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(id_error)*100:.3f}%')
    ax.set_xlabel('ID Error (log scale)')
    ax.set_ylabel('Count')
    ax.set_title('Identity Gate Error Distribution')
    ax.legend()

    # ID error by qubit
    ax = axes[0, 1]
    ax.bar(df['Qubit'], id_error*100, color='forestgreen', alpha=0.7)
    ax.axhline(np.mean(id_error)*100, color='red', linestyle='--',
               label=f'Mean: {np.mean(id_error)*100:.3f}%')
    ax.set_xlabel('Qubit Index')
    ax.set_ylabel('ID Error (%)')
    ax.set_title('Identity Gate Error by Qubit')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlim(-1, 120)

    # Comparison of single-qubit gate errors
    ax = axes[1, 0]
    gate_types = ['ID', 'RX', 'SX', 'X']
    gate_data = [
        df['ID error'].values * 100,
        df['RX error'].values * 100,
        df['√x (sx) error'].values * 100,
        df['Pauli-X error'].values * 100
    ]
    bp = ax.boxplot(gate_data, labels=gate_types, patch_artist=True)
    colors = ['forestgreen', 'steelblue', 'darkorange', 'crimson']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Error (%)')
    ax.set_title('Single-Qubit Gate Error Comparison')
    ax.set_yscale('log')

    # ID error vs T1 (expect correlation)
    ax = axes[1, 1]
    ax.scatter(df['T1 (us)'], id_error*100, alpha=0.6, c='forestgreen', s=30)
    ax.set_xlabel('T1 (μs)')
    ax.set_ylabel('ID Error (%)')
    ax.set_title('ID Error vs T1 (expect inverse correlation)')
    ax.set_yscale('log')

    corr = df['T1 (us)'].corr(df['ID error'])
    ax.text(0.95, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_connectivity(pdf: PdfPages, df: pd.DataFrame):
    """Create device connectivity visualization."""
    fig, ax = plt.subplots(figsize=(11, 8.5))

    edge_df = create_edge_dataframe(df)
    connections = count_qubit_connections(df)

    # Arrange qubits in grid
    n_cols = 12
    positions = {}
    for i in range(len(df)):
        row = i // n_cols
        col = i % n_cols
        positions[i] = (col * 1.2, -row * 1.2)

    # Draw edges colored by CZ error
    for _, row in edge_df.iterrows():
        q1, q2 = int(row['qubit1']), int(row['qubit2'])
        if q1 in positions and q2 in positions:
            x = [positions[q1][0], positions[q2][0]]
            y = [positions[q1][1], positions[q2][1]]
            # Color by error (log scale)
            error = row['cz_error']
            color_val = np.log10(error) - np.log10(edge_df['cz_error'].min())
            color_val /= (np.log10(edge_df['cz_error'].max()) - np.log10(edge_df['cz_error'].min()))
            ax.plot(x, y, color=plt.cm.Reds(color_val), linewidth=2, alpha=0.7)

    # Draw nodes
    for qubit, pos in positions.items():
        n_conn = connections.get(qubit, 0)
        if n_conn == 4:
            color = 'lightblue'
            marker = 'o'
        elif n_conn == 3:
            color = 'lightyellow'
            marker = 's'
        else:  # 2 connections
            color = 'lightcoral'
            marker = '^'
        ax.scatter(pos[0], pos[1], c=color, s=200, marker=marker, edgecolors='black', linewidth=1, zorder=5)
        ax.text(pos[0], pos[1], str(qubit), ha='center', va='center', fontsize=6, zorder=6)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('IBM Miami Heavy-Hex Topology\n(Edge color: CZ error, Node shape: connectivity)',
                 fontsize=12, fontweight='bold')

    # Legend
    ax.scatter([], [], c='lightblue', s=100, marker='o', edgecolors='black', label='4 neighbors (interior)')
    ax.scatter([], [], c='lightyellow', s=100, marker='s', edgecolors='black', label='3 neighbors (edge)')
    ax.scatter([], [], c='lightcoral', s=100, marker='^', edgecolors='black', label='2 neighbors (corner)')
    ax.legend(loc='upper right', fontsize=9)

    # Colorbar for edges
    sm = plt.cm.ScalarMappable(cmap='Reds',
                                norm=plt.Normalize(vmin=np.log10(edge_df['cz_error'].min()),
                                                   vmax=np.log10(edge_df['cz_error'].max())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='log10(CZ Error)')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_correlation_matrix(pdf: PdfPages, df: pd.DataFrame):
    """Create correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Add average CZ error per qubit
    avg_cz = {}
    for _, row in df.iterrows():
        qubit = row['Qubit']
        cz_dict = parse_cz_errors(row['CZ error'])
        if cz_dict:
            avg_cz[qubit] = np.mean(list(cz_dict.values()))
        else:
            avg_cz[qubit] = np.nan
    df_corr = df.copy()
    df_corr['Avg CZ Error'] = df_corr['Qubit'].map(avg_cz)

    cols = ['T1 (us)', 'T2 (us)', 'Readout assignment error',
            'Prob meas0 prep1', 'Prob meas1 prep0', 'ID error', 'Avg CZ Error']

    corr = df_corr[cols].corr()

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson Correlation')

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    # Annotate
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

    ax.set_title('Parameter Correlation Matrix\n(Strong correlations indicate shared physical mechanisms)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_outlier_summary(pdf: PdfPages, df: pd.DataFrame):
    """Create outlier summary page."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle('Outlier Detection Summary (IQR Method: >Q3 + 1.5×IQR)',
                 fontsize=12, fontweight='bold')

    metrics = [
        ('T1 (us)', df['T1 (us)'].values, 'T1 (μs)', 'steelblue'),
        ('T2 (us)', df['T2 (us)'].values, 'T2 (μs)', 'darkorange'),
        ('Readout', df['Readout assignment error'].values*100, 'Readout Error (%)', 'mediumpurple'),
        ('ID Error', df['ID error'].values*100, 'ID Error (%)', 'forestgreen'),
        ('CZ Error', get_cz_errors_array(df)*100, 'CZ Error (%)', 'crimson'),
        ('Gate Length', get_gate_lengths_array(df), 'Gate Length (ns)', 'teal'),
    ]

    for ax, (name, data, xlabel, color) in zip(axes.flatten(), metrics):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        low_thresh = q1 - 1.5 * iqr
        high_thresh = q3 + 1.5 * iqr

        outliers = (data < low_thresh) | (data > high_thresh)
        n_outliers = np.sum(outliers)

        ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color=color)
        ax.axvline(high_thresh, color='red', linestyle='--', linewidth=2)
        ax.axvline(low_thresh, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'{name}\n{n_outliers} outliers ({n_outliers/len(data)*100:.1f}%)', fontsize=10)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def create_best_worst_table(pdf: PdfPages, df: pd.DataFrame):
    """Create table of best and worst qubits."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))

    # Calculate composite score
    df_score = df.copy()
    avg_cz = {}
    for _, row in df.iterrows():
        qubit = row['Qubit']
        cz_dict = parse_cz_errors(row['CZ error'])
        if cz_dict:
            avg_cz[qubit] = np.mean(list(cz_dict.values()))
        else:
            avg_cz[qubit] = 0
    df_score['Avg CZ Error'] = df_score['Qubit'].map(avg_cz)

    from scipy import stats as sp_stats
    metrics = ['Readout assignment error', 'ID error', 'Avg CZ Error']
    for m in metrics:
        df_score[f'{m}_z'] = sp_stats.zscore(df_score[m])
    df_score['T1_inv_z'] = sp_stats.zscore(1 / (df_score['T1 (us)'] + 1e-10))
    df_score['T2_inv_z'] = sp_stats.zscore(1 / (df_score['T2 (us)'] + 1e-10))

    df_score['Score'] = (df_score['Readout assignment error_z'] +
                         df_score['ID error_z'] +
                         df_score['Avg CZ Error_z'] +
                         df_score['T1_inv_z'] +
                         df_score['T2_inv_z'])

    # Worst qubits
    ax = axes[0]
    ax.axis('off')
    worst = df_score.nlargest(15, 'Score')[['Qubit', 'T1 (us)', 'T2 (us)',
                                             'Readout assignment error', 'ID error', 'Avg CZ Error']]

    table_data = [['Qubit', 'T1 (μs)', 'T2 (μs)', 'Readout', 'ID Err', 'CZ Err']]
    for _, row in worst.iterrows():
        table_data.append([
            f"{int(row['Qubit'])}",
            f"{row['T1 (us)']:.0f}",
            f"{row['T2 (us)']:.0f}",
            f"{row['Readout assignment error']*100:.2f}%",
            f"{row['ID error']*100:.3f}%",
            f"{row['Avg CZ Error']*100:.2f}%"
        ])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for i in range(6):
        table[(0, i)].set_facecolor('#C00000')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax.set_title('15 WORST Qubits\n(Highest composite error score)', fontsize=11, fontweight='bold', color='darkred')

    # Best qubits
    ax = axes[1]
    ax.axis('off')
    best = df_score.nsmallest(15, 'Score')[['Qubit', 'T1 (us)', 'T2 (us)',
                                             'Readout assignment error', 'ID error', 'Avg CZ Error']]

    table_data = [['Qubit', 'T1 (μs)', 'T2 (μs)', 'Readout', 'ID Err', 'CZ Err']]
    for _, row in best.iterrows():
        table_data.append([
            f"{int(row['Qubit'])}",
            f"{row['T1 (us)']:.0f}",
            f"{row['T2 (us)']:.0f}",
            f"{row['Readout assignment error']*100:.2f}%",
            f"{row['ID error']*100:.3f}%",
            f"{row['Avg CZ Error']*100:.2f}%"
        ])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for i in range(6):
        table[(0, i)].set_facecolor('#00B050')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax.set_title('15 BEST Qubits\n(Lowest composite error score)', fontsize=11, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def create_statistics_tables(pdf: PdfPages, df: pd.DataFrame):
    """Create comprehensive statistics tables."""
    cz_errors = get_cz_errors_array(df)
    gate_lengths = get_gate_lengths_array(df)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    # Comprehensive statistics table
    stats_data = [
        ['Parameter', 'Count', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Outliers'],
    ]

    params = [
        ('T1 (μs)', df['T1 (us)'].values),
        ('T2 (μs)', df['T2 (us)'].values),
        ('Readout Error (%)', df['Readout assignment error'].values * 100),
        ('P(0|1) (%)', df['Prob meas0 prep1'].values * 100),
        ('P(1|0) (%)', df['Prob meas1 prep0'].values * 100),
        ('ID Error (%)', df['ID error'].values * 100),
        ('CZ Error (%)', cz_errors * 100),
        ('Gate Length (ns)', gate_lengths.astype(float)),
    ]

    for name, data in params:
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        outliers = np.sum((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr))

        stats_data.append([
            name,
            f"{len(data)}",
            f"{np.mean(data):.4f}",
            f"{np.std(data):.4f}",
            f"{np.median(data):.4f}",
            f"{np.min(data):.4f}",
            f"{np.max(data):.4f}",
            f"{outliers}"
        ])

    table = ax.table(cellText=stats_data, loc='center', cellLoc='center',
                     colWidths=[0.18, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)

    for i in range(8):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Comprehensive Statistical Summary\n(Outliers detected via IQR method)',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(csv_path: str, output_path: str = 'ibm_miami_report.pdf'):
    """
    Generate complete PDF report.

    Parameters
    ----------
    csv_path : str
        Path to calibration CSV file.
    output_path : str
        Output PDF file path.
    """
    print(f"Loading data from {csv_path}...")
    df = load_calibration_data(csv_path)

    print(f"Generating report: {output_path}")

    with PdfPages(output_path) as pdf:
        print("  Creating title page...")
        create_title_page(pdf, csv_path)

        print("  Creating table of contents...")
        create_toc_page(pdf)

        print("  Creating executive summary...")
        create_executive_summary(pdf, df)

        print("  Creating device overview...")
        create_text_page(pdf, "Device Overview", DEVICE_DESCRIPTION)

        print("  Creating parameter definitions...")
        create_parameter_pages(pdf)

        print("  Creating coherence analysis...")
        plot_coherence_analysis(pdf, df)

        print("  Creating CZ gate analysis...")
        plot_cz_analysis(pdf, df)

        print("  Creating readout analysis...")
        plot_readout_analysis(pdf, df)

        print("  Creating single-qubit analysis...")
        plot_single_qubit_analysis(pdf, df)

        print("  Creating connectivity visualization...")
        plot_connectivity(pdf, df)

        print("  Creating correlation matrix...")
        plot_correlation_matrix(pdf, df)

        print("  Creating outlier summary...")
        plot_outlier_summary(pdf, df)

        print("  Creating best/worst qubit tables...")
        create_best_worst_table(pdf, df)

        print("  Creating statistics tables...")
        create_statistics_tables(pdf, df)

    print(f"\nReport generated: {output_path}")
    return output_path


if __name__ == "__main__":
    import glob
    import sys

    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found")
        sys.exit(1)

    generate_report(csv_files[0])
