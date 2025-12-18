"""
Data models for quantum computing platform analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class PlatformType(Enum):
    """Quantum computing platform modalities."""
    SUPERCONDUCTING = "superconducting"
    TRAPPED_ION = "trapped_ion"
    NEUTRAL_ATOM = "neutral_atom"
    PHOTONIC = "photonic"
    SILICON = "silicon"
    NV_CENTER = "nv_center"
    NMR = "nmr"
    TOPOLOGICAL = "topological"
    QUANTUM_ANNEALING = "quantum_annealing"
    HYBRID = "hybrid"
    OTHER = "other"


class Connectivity(Enum):
    """Qubit connectivity topologies."""
    ALL_TO_ALL = "all_to_all"
    LINEAR = "linear"
    GRID = "grid"
    HEAVY_HEX = "heavy_hex"
    PEGASUS = "pegasus"
    ZEPHYR = "zephyr"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class ErrorCorrectionStatus(Enum):
    """Error correction implementation status."""
    NONE = "none"
    RESEARCH = "research"
    DEMONSTRATED = "demonstrated"
    INTEGRATED = "integrated"
    PRODUCTION = "production"


@dataclass
class CoherenceTimes:
    """Coherence time measurements."""
    t1_us: Optional[float] = None  # T1 in microseconds
    t2_us: Optional[float] = None  # T2 in microseconds
    t2_star_us: Optional[float] = None  # T2* in microseconds


@dataclass
class GateFidelities:
    """Gate fidelity measurements."""
    single_qubit: Optional[float] = None  # Percentage (0-100)
    two_qubit: Optional[float] = None  # Percentage (0-100)
    measurement: Optional[float] = None  # Percentage (0-100)
    state_prep: Optional[float] = None  # Percentage (0-100)


@dataclass
class DiVincenzoScores:
    """DiVincenzo criteria scores (0-20 each, 0-100 total)."""
    scalable_system: float = 0.0  # Scalable physical system with characterized qubits
    initialization: float = 0.0  # Qubit initialization capability
    coherence_ratio: float = 0.0  # Decoherence times >> gate times
    universal_gates: float = 0.0  # Universal gate set
    measurement: float = 0.0  # Measurement fidelity
    
    @property
    def total(self) -> float:
        """Calculate total DiVincenzo score."""
        return (self.scalable_system + self.initialization + 
                self.coherence_ratio + self.universal_gates + self.measurement)


@dataclass
class QuantumReadinessIndex:
    """Quantum Readiness Index components (0-100 total)."""
    hardware_maturity: float = 0.0  # Max 40 points
    error_correction: float = 0.0  # Max 25 points
    divincenzo: float = 0.0  # Max 25 points (scaled from DiVincenzo total)
    application_demos: float = 0.0  # Max 10 points
    
    @property
    def total(self) -> float:
        """Calculate total QRI score."""
        return (self.hardware_maturity + self.error_correction + 
                self.divincenzo + self.application_demos)


@dataclass
class QuantumAdvantageGap:
    """Quantum advantage gap analysis for options pricing."""
    current_logical_qubits: int = 0
    required_logical_qubits: int = 50  # Target: 50-100
    logical_qubit_gap: float = 100.0  # Percentage gap
    
    current_error_rate: float = 1.0  # Current error rate
    required_error_rate: float = 1e-15  # Target: 10^-15
    error_rate_gap: float = 100.0  # Percentage gap
    
    current_circuit_depth: int = 0
    required_circuit_depth: int = 5000  # Target: 5000+ gates
    circuit_depth_gap: float = 100.0  # Percentage gap
    
    overall_gap: float = 100.0  # Weighted average gap


@dataclass
class TimelineProjection:
    """Timeline projections for quantum milestones."""
    qec_threshold_year: Optional[int] = None
    hundred_logical_qubits_year: Optional[int] = None
    options_advantage_year: Optional[int] = None
    confidence_level: str = "low"  # low, medium, high


@dataclass
class QuantumSystem:
    """Complete quantum computing system data."""
    # Basic identification
    id: str = ""
    name: str = ""
    company: str = ""
    country: str = ""
    
    # Platform characteristics
    platform_type: PlatformType = PlatformType.OTHER
    physical_qubits: int = 0
    logical_qubits: Optional[int] = None
    
    # Performance metrics
    fidelities: GateFidelities = field(default_factory=GateFidelities)
    coherence: CoherenceTimes = field(default_factory=CoherenceTimes)
    quantum_volume: Optional[int] = None
    clops: Optional[int] = None  # Circuit Layer Operations Per Second
    
    # Architecture
    connectivity: Connectivity = Connectivity.UNKNOWN
    error_correction: ErrorCorrectionStatus = ErrorCorrectionStatus.NONE
    
    # Assessment scores
    divincenzo: DiVincenzoScores = field(default_factory=DiVincenzoScores)
    qri: QuantumReadinessIndex = field(default_factory=QuantumReadinessIndex)
    advantage_gap: QuantumAdvantageGap = field(default_factory=QuantumAdvantageGap)
    timeline: TimelineProjection = field(default_factory=TimelineProjection)
    
    # Additional metadata
    year_announced: Optional[int] = None
    cloud_access: bool = False
    documentation_url: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "company": self.company,
            "country": self.country,
            "platform_type": self.platform_type.value,
            "physical_qubits": self.physical_qubits,
            "logical_qubits": self.logical_qubits,
            "fidelities": {
                "single_qubit": self.fidelities.single_qubit,
                "two_qubit": self.fidelities.two_qubit,
                "measurement": self.fidelities.measurement,
                "state_prep": self.fidelities.state_prep
            },
            "coherence": {
                "t1_us": self.coherence.t1_us,
                "t2_us": self.coherence.t2_us,
                "t2_star_us": self.coherence.t2_star_us
            },
            "quantum_volume": self.quantum_volume,
            "clops": self.clops,
            "connectivity": self.connectivity.value,
            "error_correction": self.error_correction.value,
            "divincenzo_scores": {
                "scalable_system": self.divincenzo.scalable_system,
                "initialization": self.divincenzo.initialization,
                "coherence_ratio": self.divincenzo.coherence_ratio,
                "universal_gates": self.divincenzo.universal_gates,
                "measurement": self.divincenzo.measurement,
                "total": self.divincenzo.total
            },
            "quantum_readiness_index": {
                "hardware_maturity": self.qri.hardware_maturity,
                "error_correction": self.qri.error_correction,
                "divincenzo": self.qri.divincenzo,
                "application_demos": self.qri.application_demos,
                "total": self.qri.total
            },
            "advantage_gap": {
                "current_logical_qubits": self.advantage_gap.current_logical_qubits,
                "required_logical_qubits": self.advantage_gap.required_logical_qubits,
                "logical_qubit_gap": self.advantage_gap.logical_qubit_gap,
                "current_error_rate": self.advantage_gap.current_error_rate,
                "required_error_rate": self.advantage_gap.required_error_rate,
                "error_rate_gap": self.advantage_gap.error_rate_gap,
                "current_circuit_depth": self.advantage_gap.current_circuit_depth,
                "required_circuit_depth": self.advantage_gap.required_circuit_depth,
                "circuit_depth_gap": self.advantage_gap.circuit_depth_gap,
                "overall_gap": self.advantage_gap.overall_gap
            },
            "timeline_projection": {
                "qec_threshold_year": self.timeline.qec_threshold_year,
                "hundred_logical_qubits_year": self.timeline.hundred_logical_qubits_year,
                "options_advantage_year": self.timeline.options_advantage_year,
                "confidence_level": self.timeline.confidence_level
            },
            "year_announced": self.year_announced,
            "cloud_access": self.cloud_access,
            "documentation_url": self.documentation_url,
            "notes": self.notes
        }
