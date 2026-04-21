# Quantum Hardware Systems
Characterize quantum hardware

| Company | Countries | Qubits | Modality | QEC | Status |
|---------|-----------|--------|----------|-----|--------|
| **Atom** (AC1000) | USA | 1225 | Neut-atom | High | Active |
| **IBM** (Heron r2/Condor) | USA, DEU, JPN, ESP, CAN, KOR | 1121 | SC | High | Active |
| **Pasqal** (Orion) | FRA, DEU, CAN, SAU | >1000 (200 deployed) | Neut-atom | High | Active |
| **Google** (Willow) | USA | 105 | SC | High | Active |
| **IonQ** (Tempo) | USA, KOR | 100 | Trap-ion | High | Active |
| **Quantinuum** (Helios) | USA, GBR | 98 + 48 logical | Trap-ion | High | Active |
| **Xanadu** (Aurora) | CAN | 12 (modular) | Photon | High | Active |
| **PsiQuantum** (Omega) | USA, AUS | N/A | Photon | High | Active |
| **QuEra** (Aquila+) | USA | 3000+ | Neut-atom | M-H | Active |
| **planqc** | DEU | ~100-200 (1000 plan) | Neut-atom | M-H | Active |
| **IQM** (Radiance) | FIN, DEU, USA, POL | 150 | SC | M-H | Active |
| **Rigetti** (Ankaa-3) | USA, GBR | 84 | SC | M-H | Active |
| **SQC** | AUS | 11 | Si-donor | M-H | Active |
| **Photonic** | CAN | Few | Spin-photon | M-H | Active |
| **Quantum Motion** (CMOS QC) | GBR | Few | Si-spin | M-H | Active |
| **D-Wave** (Adv2) | CAN, USA, DEU | ~5000 / ~4400 | QA | Med | Active |
| **QuiX** | NLD | 50-100+ modes | Photon | Med | Active |
| **ORCA** (PT-2) | GBR | ~90 | Photon | Med | Active |
| **Origin** (Wukong) | CHN | 72 | SC | Med | Active |
| **EleQtron** (MAGIC) | DEU | ~30 | Trap-ion | Med | Active |
| **AQT** (PINE/IBEX) | AUT, DEU, POL | 24 | Trap-ion | Med | Active |
| **Quandela** (Lucy/Canopus) | FRA, KOR | 12-24 | Photon | Med | Active |
| **Q-Brilliance** | AUS, DEU, USA | 2/module | NV-diamond | L-M | Active |
| **SpinQ** (Gemini) | CHN | 2-3 | NMR | Low | Active |
| **Oxford Ionics** (QUARTET) | GBR | ~10-20 | Trap-ion | Med | Acquired |
| **Baidu** (Qianshi) | CHN | ~10-36 | SC | L-M | Discontinued |
| **Alibaba** | CHN | ~11 | SC | Low | Discontinued |

## Feature Explanations

**Countries**: ISO 3-letter country codes (AUT=Austria, AUS=Australia, CAN=Canada, CHN=China, DEU=Germany, ESP=Spain, FIN=Finland, FRA=France, GBR=United Kingdom, JPN=Japan, KOR=South Korea, NLD=Netherlands, POL=Poland, SAU=Saudi Arabia, USA=United States)

**Qubits**: Maximum public physical qubits or modes

**Modality**: 
- SC = Superconducting
- Trap-ion = Trapped ions
- Neut-atom = Neutral atoms
- Photon = Photonic
- Photon-CV = Photonic continuous variable
- Si-spin = Silicon spin qubits
- Si-donor = Silicon donor qubits
- NV-diamond = NV-center diamond
- QA = Quantum annealing
- Spin-photon = Spin-photon hybrid
- NMR = Nuclear magnetic resonance

**QEC**: Scalability to Quantum Error Correction (Low/Med/High or L-M/M-H for ranges)

**Status**: Active = currently operating/available; Acquired = company acquired by another; Discontinued = quantum program shut down

---

*Data last verified: 2026-04-20. Updated weekly via GitHub Actions — see [scripts/update_hardware_data.py](scripts/update_hardware_data.py) and [data/hardware_data.json](data/hardware_data.json) for sources and methodology.*
