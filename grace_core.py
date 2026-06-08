from dataclasses import dataclass

@dataclass
class GraceState:
    N: float = 70
    S: float = 70
    R: float = 70
    C: float = 70
    E: float = 70
    A: float = 70

def clamp(x, low=0, high=100):
    return max(low, min(high, x))

def calculate_grace_index(state: GraceState) -> float:
    return (
        0.22 * state.N +
        0.14 * state.S +
        0.20 * state.R +
        0.16 * state.C +
        0.13 * state.E +
        0.15 * state.A
    )

def get_zone(score: float) -> str:
    if score >= 85:
        return "Flourishing"
    elif score >= 70:
        return "Stable"
    elif score >= 50:
        return "Vulnerability"
    elif score >= 30:
        return "Disruption"
    return "Fragmentation Risk"

def apply_effect(state: GraceState, effect: dict) -> GraceState:
    return GraceState(
        N=clamp(state.N + effect.get("N", 0)),
        S=clamp(state.S + effect.get("S", 0)),
        R=clamp(state.R + effect.get("R", 0)),
        C=clamp(state.C + effect.get("C", 0)),
        E=clamp(state.E + effect.get("E", 0)),
        A=clamp(state.A + effect.get("A", 0)),
    )