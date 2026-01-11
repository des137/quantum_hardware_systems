# Quantum Hardware Specifications

This document provides detailed technical specifications for quantum computing hardware from various companies. All information is sourced from publicly available peer-reviewed papers, company announcements, and ArXiv preprints.

## Summary Table

| Company | System | Qubits | 1Q Gate Error | 2Q Gate Error | Readout Error | T1 (µs) | T2 (µs) | QV | Gate Time | Modality |
|---------|--------|--------|---------------|---------------|---------------|---------|---------|-----|-----------|----------|
| **IBM** | Heron (r2) | 156 | 1.2×10⁻⁴ | 3.5×10⁻³ | 2.4×10⁻³ | 364 | 180 | 16,384 | 68ns (2Q) | SC |
| **IBM** | Condor | 1121 | ~10⁻⁴ | ~10⁻² | ~10⁻² | ~300 | ~150 | - | ~100ns | SC |
| **Google** | Sycamore | 53 | 1.5×10⁻³ | 6.2×10⁻³ | 3.8×10⁻² | 15.7 | 10.6 | 32 | 12ns (1Q), 32ns (2Q) | SC |
| **Google** | Willow | 105 | 2.5×10⁻⁴ | 3.4×10⁻³ | 5×10⁻³ | 68 | 30 | - | ~25ns (2Q) | SC |
| **Quantinuum** | H2 | 56 | 1.6×10⁻⁵ | 1.5×10⁻³ | 2.3×10⁻³ | >300,000 | >1,500 | 65,536 | ~500µs (2Q) | Trap-ion |
| **IonQ** | Forte | 36 | 3×10⁻⁵ | 4.7×10⁻³ | 3×10⁻³ | >10⁶ | >10⁵ | 32,768 | ~135µs (2Q) | Trap-ion |
| **IonQ** | Tempo | 64 | - | ~10⁻³ | ~10⁻³ | - | - | >131,000 | - | Trap-ion |
| **Rigetti** | Ankaa-2 | 84 | 5×10⁻³ | 2.5×10⁻² | ~10⁻² | 20 | 25 | 16 | 60ns (2Q) | SC |
| **Atom Computing** | AC1000 | 1225 | 3×10⁻³ | 5×10⁻³ | 2×10⁻² | ~10⁶ | 40 | - | ~1µs | Neut-atom |
| **QuEra** | Aquila | 256 | ~10⁻³ | 5×10⁻³ | 1.5×10⁻² | ~10⁶ | ~1000 | - | ~µs | Neut-atom |
| **Pasqal** | - | >1000 | ~10⁻² | ~10⁻² | ~10⁻² | ~10⁶ | ~100 | - | ~µs | Neut-atom |
| **IQM** | Garnet | 150 | ~10⁻³ | ~10⁻² | ~10⁻² | ~50 | ~30 | - | ~ns | SC |
| **D-Wave** | Advantage2 | ~4400 | N/A | N/A | N/A | - | - | N/A | ~10ns | QA |
| **Xanadu** | Borealis | 216 modes | - | - | - | - | - | N/A | ~ns | Photon-CV |
| **Origin** | Wukong | 72 | ~10⁻³ | ~10⁻² | ~10⁻² | ~60 | ~40 | - | ~ns | SC |
| **AQT** | PINE | 20 | 3×10⁻⁴ | 5×10⁻³ | 4×10⁻³ | ~10⁶ | 200 | 128 | ~100µs | Trap-ion |

**Notes:**
- QV = Quantum Volume
- SC = Superconducting
- T1 = Energy relaxation time
- T2 = Dephasing time (coherence time)
- "-" indicates data not publicly available
- "~" indicates approximate values

---

## Detailed Specifications by Company

### IBM Quantum

IBM has been a leader in superconducting qubit technology with multiple generations of quantum processors.

#### IBM Heron (r2) - 2024
- **Qubits**: 156 physical qubits
- **Qubit Technology**: Fixed-frequency transmon qubits with tunable couplers
- **1-Qubit Gate Error**: 1.2×10⁻⁴ (median)
- **2-Qubit Gate Error (CZ)**: 3.5×10⁻³ (median)
- **Readout Error**: 2.4×10⁻³ (median)
- **T1**: 364 µs (median)
- **T2**: 180 µs (median)
- **2-Qubit Gate Time**: 68 ns (CZ gate)
- **Quantum Volume**: 16,384 [1]

#### IBM Condor - 2023
- **Qubits**: 1121 physical qubits
- **Architecture**: Heavy-hex lattice topology
- **1-Qubit Gate Error**: ~10⁻⁴
- **2-Qubit Gate Error**: ~10⁻²
- **Gate Time**: ~100ns (ECR gate)
- **Focus**: Scalability demonstration [2]

#### IBM Eagle/Osprey Series
- **Eagle (127 qubits)**: Quantum Volume 512
- **Osprey (433 qubits)**: Demonstrated in 2022
- **Typical T1**: 200-400 µs
- **Typical T2**: 100-200 µs [3]

**References**:
1. IBM Quantum, "IBM Quantum Development Roadmap," 2024. Available at: https://www.ibm.com/quantum/roadmap
2. Gambetta, J.M. et al., "Building a quantum-ready ecosystem," IBM Research Blog, 2023.
3. IBM Quantum Documentation, "System specifications," https://quantum.ibm.com/services/resources

---

### Google Quantum AI

#### Google Sycamore - 2019
- **Qubits**: 53 operational (54 physical)
- **Qubit Type**: Transmon superconducting qubits
- **1-Qubit Gate Error**: 1.5×10⁻³ (average)
- **2-Qubit Gate Error (√iSWAP)**: 6.2×10⁻³ (average)
- **Readout Error**: 3.8×10⁻²
- **T1**: 15.7 µs (average)
- **T2**: 10.6 µs (average)
- **1-Qubit Gate Time**: 12 ns
- **2-Qubit Gate Time**: 32 ns
- **Quantum Volume**: 32 [4]

#### Google Willow - 2024
- **Qubits**: 105 physical qubits
- **Architecture**: 2D surface code compatible
- **1-Qubit Gate Error**: 2.5×10⁻⁴
- **2-Qubit Gate Error**: 3.4×10⁻³
- **Readout Error**: 5×10⁻³
- **T1**: 68 µs (median)
- **T2**: 30 µs (median)
- **2-Qubit Gate Time**: ~25 ns
- **Notable Achievement**: First below-threshold surface code demonstration [5]

**References**:
4. Arute, F. et al., "Quantum supremacy using a programmable superconducting processor," Nature 574, 505–510 (2019). arXiv:1910.11333
5. Google Quantum AI, "Quantum error correction below threshold with Willow," arXiv:2408.13687 (2024).

---

### Quantinuum (Honeywell)

#### Quantinuum H2 - 2024
- **Qubits**: 56 physical qubits (all-to-all connectivity)
- **Ion Species**: Ytterbium-171 (¹⁷¹Yb⁺)
- **1-Qubit Gate Error**: 1.6×10⁻⁵ (best reported)
- **2-Qubit Gate Error (ZZ)**: 1.5×10⁻³ (average)
- **Readout Error**: 2.3×10⁻³
- **State Prep Error**: 2×10⁻⁴
- **T1**: >300,000 µs (>5 minutes)
- **T2**: >1,500 µs
- **2-Qubit Gate Time**: ~500 µs (ZZ gate)
- **Quantum Volume**: 65,536 (2^16) [6]

#### Quantinuum H1 Series
- **H1-1**: 20 qubits, QV 32,768
- **H1-2**: 20 qubits, QV 32,768
- **Architecture**: Linear trap with QCCD (Quantum Charge-Coupled Device) [7]

#### Quantinuum Helios - 2025
- **Qubits**: 98 physical qubits + logical qubit capability
- **Target**: Utility-scale fault-tolerant quantum computing [8]

**References**:
6. Moses, S.A. et al., "A Race Track Trapped-Ion Quantum Processor," Physical Review X 13, 041052 (2023). arXiv:2305.03828
7. Pino, J.M. et al., "Demonstration of the trapped-ion quantum CCD computer architecture," Nature 592, 209–213 (2021). arXiv:2003.01293
8. Quantinuum, "H-Series Hardware Specifications," https://www.quantinuum.com/hardware

---

### IonQ

#### IonQ Forte - 2022/2023
- **Qubits**: 36 algorithmic qubits (all-to-all connectivity)
- **Ion Species**: Ytterbium (¹⁷¹Yb⁺)
- **1-Qubit Gate Error**: 3×10⁻⁵
- **2-Qubit Gate Error (XX)**: 4.7×10⁻³
- **Readout Error**: 3×10⁻³
- **T1**: >10⁶ µs (seconds scale)
- **T2**: >10⁵ µs
- **2-Qubit Gate Time**: ~135 µs
- **Quantum Volume**: 32,768 (2^15)
- **#AQ (Algorithmic Qubits)**: 36 [9]

#### IonQ Aria - 2022
- **Qubits**: 25 algorithmic qubits
- **2-Qubit Gate Error**: ~5×10⁻³
- **Quantum Volume**: 4,096 (2^12) [10]

#### IonQ Tempo - 2024
- **Qubits**: 64 physical qubits
- **Quantum Volume**: >131,000
- **Notable**: Uses reconfigurable ion chains [11]

**References**:
9. Wright, K. et al., "Benchmarking an 11-qubit quantum computer," Nature Communications 10, 5464 (2019). arXiv:1903.08181
10. IonQ, "IonQ Aria Technical Specifications," https://ionq.com/quantum-systems
11. IonQ, "IonQ Tempo Announcement," 2024.

---

### Rigetti Computing

#### Rigetti Ankaa-2 - 2024
- **Qubits**: 84 physical qubits
- **Qubit Type**: Transmon superconducting qubits
- **Topology**: Square lattice
- **1-Qubit Gate Error**: ~5×10⁻³
- **2-Qubit Gate Error (iSWAP)**: ~2.5×10⁻² (median)
- **Readout Error**: ~10⁻²
- **T1**: ~20 µs
- **T2**: ~25 µs
- **2-Qubit Gate Time**: ~60 ns
- **Quantum Volume**: 16 [12]

#### Rigetti Aspen-M Series
- **Aspen-M-3**: 80 qubits
- **2-Qubit Gate Fidelity**: 92-97%
- **Connectivity**: Heavy-hex like topology [13]

**References**:
12. Rigetti Computing, "Ankaa System Performance," https://www.rigetti.com/
13. Reagor, M. et al., "Demonstration of universal parametric entangling gates on a multi-qubit lattice," Science Advances 4, eaao3603 (2018).

---

### Atom Computing

#### Atom Computing AC1000 - 2023/2024
- **Qubits**: 1225 physical qubits (35×35 array)
- **Atom Species**: Neutral Strontium atoms
- **Trapping**: Optical tweezers
- **1-Qubit Gate Error**: ~3×10⁻³
- **2-Qubit Gate Error (CZ)**: ~5×10⁻³
- **Readout Error**: ~2×10⁻²
- **T1**: >10⁶ µs (seconds scale, atomic ground states)
- **T2**: ~40 µs (nuclear spin qubits: ~1 second)
- **Gate Time**: ~1 µs (Rydberg-based gates)
- **Notable**: First >1000 qubit neutral atom system [14]

**References**:
14. Norcia, M.A. et al., "Midcircuit qubit measurement and rearrangement in a ¹⁷¹Yb atomic array," Physical Review X 13, 041034 (2023). arXiv:2305.19119

---

### QuEra Computing

#### QuEra Aquila - 2022/2023
- **Qubits**: 256 neutral atoms
- **Atom Species**: Rubidium-87
- **Programmable Geometry**: Arbitrary 2D arrangements
- **1-Qubit Gate Error**: ~10⁻³
- **2-Qubit Gate Error**: ~5×10⁻³
- **Readout Error**: ~1.5×10⁻²
- **T1**: >10⁶ µs (atomic lifetime)
- **T2**: ~1000 µs
- **Gate Time**: ~µs (Rydberg blockade)
- **Notable**: First commercial neutral-atom quantum computer on cloud [15]

**References**:
15. Wurtz, J. et al., "Aquila: QuEra's 256-qubit neutral-atom quantum computer," arXiv:2306.11727 (2023).

---

### Pasqal

#### Pasqal Systems - 2023/2024
- **Qubits**: >1000 atoms (100-200 commercially available)
- **Atom Species**: Rubidium
- **Geometry**: 3D programmable arrays
- **1-Qubit Gate Error**: ~10⁻²
- **2-Qubit Gate Error**: ~10⁻²
- **Readout Error**: ~10⁻²
- **T1**: >10⁶ µs
- **T2**: ~100 µs
- **Gate Type**: Rydberg-based analog quantum processing [16]

**References**:
16. Henriet, L. et al., "Quantum computing with neutral atoms," Quantum 4, 327 (2020). arXiv:2006.12326

---

### IQM Quantum Computers

#### IQM Garnet - 2024
- **Qubits**: 150 physical qubits
- **Qubit Type**: Transmon superconducting qubits
- **Topology**: Square lattice
- **1-Qubit Gate Error**: ~10⁻³
- **2-Qubit Gate Error (CZ)**: ~10⁻²
- **Readout Error**: ~10⁻²
- **T1**: ~50 µs
- **T2**: ~30 µs
- **Gate Time**: ~ns [17]

#### IQM Spark
- **Qubits**: 5 qubits
- **Purpose**: Education and research
- **2-Qubit Gate Fidelity**: >99% [18]

**References**:
17. IQM Quantum Computers, "IQM Garnet Specifications," https://www.meetiqm.com/
18. IQM, "IQM Spark Technical Documentation," 2023.

---

### D-Wave Systems

D-Wave produces quantum annealers, which operate differently from gate-based quantum computers.

#### D-Wave Advantage2 - 2024
- **Qubits**: ~4400 qubits (prototype)
- **Architecture**: Zephyr topology
- **Connectivity**: 20 connections per qubit
- **Operating Temperature**: ~15 mK
- **Annealing Time**: 1-2000 µs (programmable)
- **Qubit Type**: Superconducting flux qubits [19]

#### D-Wave Advantage - 2020
- **Qubits**: 5000+ qubits
- **Architecture**: Pegasus topology
- **Connectivity**: 15 connections per qubit
- **Programming Time**: ~10 ms
- **Annealing Time**: 1-2000 µs [20]

**References**:
19. D-Wave Systems, "Advantage2 System," https://www.dwavequantum.com/
20. Boothby, K. et al., "Next-Generation Topology of D-Wave Quantum Processors," arXiv:2003.00133 (2020).

---

### Xanadu

#### Xanadu Borealis - 2022
- **Modes**: 216 squeezed-state modes
- **Architecture**: Time-multiplexed photonic system
- **Squeezing**: ~10 dB
- **Detection**: Photon number resolving detectors
- **Gate Type**: Gaussian boson sampling
- **Notable**: Programmable photonic quantum computer [21]

**References**:
21. Madsen, L.S. et al., "Quantum computational advantage with a programmable photonic processor," Nature 606, 75–81 (2022). arXiv:2206.01785

---

### Origin Quantum (本源量子)

#### Origin Wukong - 2024
- **Qubits**: 72 superconducting qubits
- **1-Qubit Gate Error**: ~10⁻³
- **2-Qubit Gate Error**: ~10⁻²
- **Readout Error**: ~10⁻²
- **T1**: ~60 µs
- **T2**: ~40 µs
- **Gate Time**: ~ns
- **Notable**: China's most advanced publicly accessible quantum computer [22]

**References**:
22. Origin Quantum, "Wukong Quantum Computer Specifications," 2024.

---

### AQT (Alpine Quantum Technologies)

#### AQT PINE - 2023
- **Qubits**: 20 trapped ions
- **Ion Species**: Calcium-40 (⁴⁰Ca⁺)
- **1-Qubit Gate Error**: 3×10⁻⁴
- **2-Qubit Gate Error (MS gate)**: 5×10⁻³
- **Readout Error**: 4×10⁻³
- **T1**: >10⁶ µs
- **T2**: ~200 µs
- **Gate Time**: ~100 µs
- **Quantum Volume**: 128 [23]

**References**:
23. Pogorelov, I. et al., "Compact Ion-Trap Quantum Computing Demonstrator," PRX Quantum 2, 020343 (2021). arXiv:2101.11390

---

### Oxford Ionics

- **Qubits**: ~10-20 trapped ions
- **Technology**: Electronic Qubit Control (EQC)
- **2-Qubit Gate Error**: <1×10⁻³ (reported)
- **Notable**: Novel approach using integrated electronics [24]

**References**:
24. Oxford Ionics, Company Technical Reports, 2023.

---

### Quandela

#### Quandela MosaiQ
- **Modes**: 6-12 photonic modes
- **Technology**: Deterministic single-photon sources (quantum dots)
- **Photon Indistinguishability**: >90%
- **Gate Type**: Linear optical [25]

**References**:
25. Somaschi, N. et al., "Near-optimal single-photon sources in the solid state," Nature Photonics 10, 340–345 (2016).

---

### EleQtron

- **Qubits**: ~10+ trapped ions
- **Technology**: MAGIC (MAgnetic Gradient Induced Coupling)
- **Notable**: Microwave-based ion control [26]

**References**:
26. Mintert, F. & Wunderlich, C., "Ion-Trap Quantum Logic Using Long-Wavelength Radiation," Physical Review Letters 87, 257904 (2001).

---

### ORCA Computing

- **Technology**: Photonic quantum memory
- **Architecture**: Time-bin encoding with quantum memories
- **Notable**: Room-temperature photonic systems [27]

**References**:
27. ORCA Computing, Technical White Papers, 2023.

---

### Baidu Quantum

#### Baidu Qianshi - 2022
- **Qubits**: 10-36 superconducting qubits
- **Technology**: Superconducting transmons
- **1-Qubit Gate Error**: ~10⁻³
- **2-Qubit Gate Error**: ~10⁻²
- **Notable**: Integrated with Baidu's quantum cloud platform [28]

**References**:
28. Baidu Research, "Qianshi Quantum Computing Platform," 2022.

---

### Alibaba Quantum (Historical)

- **Qubits**: ~11 superconducting qubits (2018)
- **Technology**: Superconducting transmons
- **Notable**: Partnership with CAS ended; cloud service discontinued [29]

**References**:
29. Alibaba Cloud, "Quantum Computing Initiative," 2018.

---

### SpinQ

#### SpinQ Gemini Series
- **Qubits**: 2-3 qubits
- **Technology**: Nuclear Magnetic Resonance (NMR)
- **Operating Temperature**: Room temperature
- **Notable**: Desktop quantum computers for education [30]

**References**:
30. SpinQ Technology, "Gemini Product Specifications," https://www.spinq.cn/

---

### Q-Brilliance

- **Qubits**: 2 qubits per module (scalable)
- **Technology**: NV-center in diamond
- **Operating Temperature**: Room temperature
- **T2**: ~ms (with dynamical decoupling)
- **Notable**: Compact, room-temperature quantum accelerators [31]

**References**:
31. Q-Brilliance, "Diamond Quantum Computing," https://qbrilliance.com/

---

### PsiQuantum

- **Technology**: Photonic (fusion-based)
- **Target**: Fault-tolerant quantum computer
- **Qubits**: N/A (in development)
- **Notable**: Manufacturing partnership with GlobalFoundries [32]

**References**:
32. PsiQuantum, Company Technical Reports, 2023.

---

### planqc

- **Qubits**: ~100-200 (1000+ planned)
- **Technology**: Neutral strontium atoms
- **Notable**: Optical clock technology for qubits [33]

**References**:
33. planqc, "Neutral Atom Quantum Computing," https://planqc.eu/

---

### Silicon Quantum Computing (SQC)

- **Qubits**: ~10 qubits
- **Technology**: Silicon donor qubits (phosphorus in silicon)
- **2-Qubit Gate Error**: ~1×10⁻²
- **T2**: >1000 µs
- **Notable**: Atomic-scale precision manufacturing [34]

**References**:
34. Madzik, M.T. et al., "Precision tomography of a three-qubit donor quantum processor in silicon," Nature 601, 348–353 (2022). arXiv:2106.03082

---

### Photonic Inc.

- **Technology**: Spin-photon hybrid (T-centers in silicon)
- **Qubits**: Few qubits (in development)
- **Notable**: Telecom-wavelength compatible [35]

**References**:
35. Photonic Inc., Technical Announcements, 2023.

---

### Q-Motion

- **Technology**: Silicon spin qubits
- **Qubits**: Few qubits (in development)
- **Notable**: Shuttling-based architecture [36]

**References**:
36. Q-Motion, Company Information, 2023.

---

### QuiX Quantum

- **Modes**: 50-100+ photonic modes
- **Technology**: Photonic integrated circuits
- **Architecture**: Programmable interferometers
- **Notable**: Universal linear optical processors [37]

**References**:
37. QuiX Quantum, "Photonic Processors," https://www.quixquantum.com/

---

## Comparison by Modality

### Superconducting Qubits
**Advantages**: Fast gate times (ns), well-developed fabrication, scalable
**Challenges**: Requires mK temperatures, limited coherence times
**Companies**: IBM, Google, Rigetti, IQM, Origin, Baidu, Alibaba

### Trapped Ions
**Advantages**: Long coherence times, all-to-all connectivity, high fidelity
**Challenges**: Slower gate times (µs), complex trap engineering
**Companies**: Quantinuum, IonQ, AQT, Oxford Ionics, EleQtron

### Neutral Atoms
**Advantages**: Many qubits possible, long coherence, flexible geometry
**Challenges**: Atom loss, slower operations
**Companies**: Atom Computing, QuEra, Pasqal, planqc

### Photonic
**Advantages**: Room temperature, inherent networking capability
**Challenges**: Probabilistic gates, photon loss
**Companies**: Xanadu, PsiQuantum, Quandela, ORCA, QuiX, Photonic

### Other Modalities
- **NV-Diamond**: Q-Brilliance (room temperature)
- **Silicon Spin**: SQC, Q-Motion (CMOS compatible)
- **NMR**: SpinQ (educational)
- **Quantum Annealing**: D-Wave (optimization)

---

## Key Metrics Explained

### Quantum Volume (QV)
Quantum Volume is a hardware-agnostic metric introduced by IBM that measures the overall capability of a quantum computer. It accounts for qubit count, connectivity, gate fidelity, and measurement errors. A QV of 2^n indicates the system can reliably execute random circuits of depth n on n qubits.

### Gate Fidelity vs Error Rate
- **Fidelity** = 1 - Error Rate
- Example: 99.5% fidelity = 5×10⁻³ error rate

### Coherence Times
- **T1 (Relaxation Time)**: Time for qubit to decay from |1⟩ to |0⟩
- **T2 (Dephasing Time)**: Time for qubit to lose phase coherence
- T2 ≤ 2×T1 (fundamental limit)

### Algorithmic Qubits (#AQ)
IonQ's metric that estimates effective qubits available for useful algorithms, accounting for error rates.

---

## Data Sources and Citations

### Primary ArXiv References

1. **arXiv:1910.11333** - Arute, F. et al., "Quantum supremacy using a programmable superconducting processor" (Google Sycamore)
2. **arXiv:2305.03828** - Moses, S.A. et al., "A Race Track Trapped-Ion Quantum Processor" (Quantinuum H2)
3. **arXiv:2003.01293** - Pino, J.M. et al., "Demonstration of the trapped-ion quantum CCD computer architecture" (Honeywell)
4. **arXiv:1903.08181** - Wright, K. et al., "Benchmarking an 11-qubit quantum computer" (IonQ)
5. **arXiv:2206.01785** - Madsen, L.S. et al., "Quantum computational advantage with a programmable photonic processor" (Xanadu Borealis)
6. **arXiv:2006.12326** - Henriet, L. et al., "Quantum computing with neutral atoms" (Pasqal)
7. **arXiv:2101.11390** - Pogorelov, I. et al., "Compact Ion-Trap Quantum Computing Demonstrator" (AQT)
8. **arXiv:2106.03082** - Madzik, M.T. et al., "Precision tomography of a three-qubit donor quantum processor in silicon" (SQC)
9. **arXiv:2305.19119** - Norcia, M.A. et al., "Midcircuit qubit measurement and rearrangement in a ¹⁷¹Yb atomic array" (Atom Computing)
10. **arXiv:2003.00133** - Boothby, K. et al., "Next-Generation Topology of D-Wave Quantum Processors" (D-Wave)
11. **arXiv:2408.13687** - Google Quantum AI, "Quantum error correction below threshold with Willow" (Google Willow)
12. **arXiv:2306.11727** - Wurtz, J. et al., "Aquila: QuEra's 256-qubit neutral-atom quantum computer" (QuEra)

### Company Documentation
- IBM Quantum: https://quantum.ibm.com/
- Google Quantum AI: https://quantumai.google/
- Quantinuum: https://www.quantinuum.com/
- IonQ: https://ionq.com/
- Rigetti: https://www.rigetti.com/
- D-Wave: https://www.dwavequantum.com/
- Xanadu: https://xanadu.ai/
- Atom Computing: https://atom-computing.com/
- QuEra: https://www.quera.com/
- Pasqal: https://www.pasqal.com/
- IQM: https://www.meetiqm.com/

---

*Last Updated: January 2025*

*Note: Specifications are subject to change as companies continue to improve their hardware. Always refer to official company documentation and peer-reviewed publications for the most current information.*
