import random
import pandas as pd

from grace_core import GraceState, calculate_grace_index, get_zone, apply_effect
from event_catalog import EVENTS
from simulator import STIMULUS_EFFECTS, choose_stimulus
from story_agent import generate_simulated_daily_story, analyze_grace_with_llm


def state_to_dict(state: GraceState) -> dict:
    return {
        "N": state.N,
        "S": state.S,
        "R": state.R,
        "C": state.C,
        "E": state.E,
        "A": state.A,
        "GRACE": round(calculate_grace_index(state), 2),
        "zone": get_zone(calculate_grace_index(state))
    }


def run_simulation_with_llm(
    profile: dict,
    days: int = 30,
    seed: int = 42,
    ollama_url: str = "http://100.112.221.112:11434",
    model: str = "gemma4"
):
    random.seed(seed)
    state = GraceState()
    rows = []

    for day in range(1, days + 1):
        event = random.choice(EVENTS)

        grace_before = state_to_dict(state)

        # 1. Event memengaruhi state
        state = apply_effect(state, event["effect"])

        # 2. LLM membuat cerita harian berdasarkan profil + event
        daily_story = generate_simulated_daily_story(
            profile=profile,
            event_name=event["name"],
            grace_before=grace_before,
            ollama_url=ollama_url,
            model=model,
        )

        # 3. LLM menganalisis cerita menjadi GRACE signal
        grace_data = analyze_grace_with_llm(
            raw_story=event["name"],
            narrative=daily_story,
            ollama_url=ollama_url,
            model=model,
        )

        # 4. Untuk quick win, gabungkan rule event dengan interpretasi LLM secara ringan
        #    Ambil rata-rata state setelah event dan skor LLM.
        state = GraceState(
            N=(state.N + grace_data["N"]) / 2,
            S=(state.S + grace_data["S"]) / 2,
            R=(state.R + grace_data["R"]) / 2,
            C=(state.C + grace_data["C"]) / 2,
            E=(state.E + grace_data["E"]) / 2,
            A=(state.A + grace_data["A"]) / 2,
        )

        # 5. Pilih stimulus
        stimulus = choose_stimulus(state, event["recommended"])

        # 6. Terapkan stimulus
        state = apply_effect(state, STIMULUS_EFFECTS[stimulus])

        score = calculate_grace_index(state)
        zone = get_zone(score)

        rows.append({
            "day": day,
            "profile": profile.get("name"),
            "event": event["name"],
            "daily_story": daily_story,
            "stimulus": stimulus,
            "N": round(state.N, 2),
            "S": round(state.S, 2),
            "R": round(state.R, 2),
            "C": round(state.C, 2),
            "E": round(state.E, 2),
            "A": round(state.A, 2),
            "GRACE": round(score, 2),
            "zone": zone,
            "values": ", ".join(grace_data.get("values", [])),
            "emotion": ", ".join(grace_data.get("emotion", [])),
            "risk_notes": grace_data.get("risk_notes", ""),
            "recommended_stimulus_llm": grace_data.get("recommended_stimulus", "")
        })

    return pd.DataFrame(rows)