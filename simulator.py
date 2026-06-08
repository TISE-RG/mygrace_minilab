import random
import pandas as pd
from grace_core import GraceState, calculate_grace_index, get_zone, apply_effect
from event_catalog import EVENTS

STIMULUS_EFFECTS = {
    "a_nar": {"N": 4, "A": 1},
    "a_rel": {"R": 5, "N": 1, "A": 2},
    "a_spi": {"S": 5, "A": 3, "N": 1},
    "a_con": {"C": 6, "N": 2, "R": 1},
    "a_com": {"E": 5, "R": 2, "C": 1},
    "a_prac": {"A": 5, "E": 1},
    "a_none": {}
}

def choose_stimulus(state: GraceState, recommended: str):
    # Quick win: gunakan rekomendasi event, tetapi cek dimensi paling rendah.
    scores = {
        "N": state.N,
        "S": state.S,
        "R": state.R,
        "C": state.C,
        "E": state.E,
        "A": state.A
    }

    lowest = min(scores, key=scores.get)

    if scores[lowest] < 60:
        return {
            "N": "a_nar",
            "S": "a_spi",
            "R": "a_rel",
            "C": "a_con",
            "E": "a_com",
            "A": "a_prac"
        }[lowest]

    return recommended

def run_simulation(days=30, seed=42):
    random.seed(seed)
    state = GraceState()
    rows = []

    for day in range(1, days + 1):
        event = random.choice(EVENTS)

        state = apply_effect(state, event["effect"])

        stimulus = choose_stimulus(state, event["recommended"])
        state = apply_effect(state, STIMULUS_EFFECTS[stimulus])

        score = calculate_grace_index(state)
        zone = get_zone(score)

        rows.append({
            "day": day,
            "event": event["name"],
            "stimulus": stimulus,
            "N": state.N,
            "S": state.S,
            "R": state.R,
            "C": state.C,
            "E": state.E,
            "A": state.A,
            "GRACE": round(score, 2),
            "zone": zone
        })

    return pd.DataFrame(rows)