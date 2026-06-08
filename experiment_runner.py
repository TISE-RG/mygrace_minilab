import os
import random
import pandas as pd

from grace_core import GraceState, calculate_grace_index, get_zone, apply_effect
from event_catalog import EVENTS
from simulator import STIMULUS_EFFECTS, choose_stimulus
from simulator_llm import run_simulation_with_llm


def count_interventions(df: pd.DataFrame) -> int:
    if "stimulus" not in df.columns:
        return 0
    return len(df[df["stimulus"] != "a_none"])


def summarize_result(df: pd.DataFrame, condition: str) -> dict:
    stable_days = len(df[df["GRACE"] >= 70])
    vulnerable_days = len(df[df["GRACE"] < 70])

    min_grace = df["GRACE"].min()
    max_grace = df["GRACE"].max()
    avg_grace = df["GRACE"].mean()
    final_grace = df["GRACE"].iloc[-1]

    intervention_count = count_interventions(df)

    return {
        "condition": condition,
        "average_grace": round(avg_grace, 2),
        "final_grace": round(final_grace, 2),
        "min_grace": round(min_grace, 2),
        "max_grace": round(max_grace, 2),
        "stable_days": stable_days,
        "vulnerable_days": vulnerable_days,
        "intervention_count": intervention_count,
        "days": len(df),
    }


def run_baseline(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Baseline:
    Event memengaruhi state.
    Tidak ada story effect.
    Tidak ada stimulus.
    """
    random.seed(seed)
    state = GraceState()
    rows = []

    for day in range(1, days + 1):
        event = random.choice(EVENTS)

        state = apply_effect(state, event["effect"])

        score = calculate_grace_index(state)
        zone = get_zone(score)

        rows.append({
            "day": day,
            "condition": "Baseline",
            "event": event["name"],
            "stimulus": "a_none",
            "N": round(state.N, 2),
            "S": round(state.S, 2),
            "R": round(state.R, 2),
            "C": round(state.C, 2),
            "E": round(state.E, 2),
            "A": round(state.A, 2),
            "GRACE": round(score, 2),
            "zone": zone,
        })

    return pd.DataFrame(rows)


def run_story_only(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Story Only:
    Event memengaruhi state.
    Lalu ada efek story/reframing ringan pada Narrative Continuity dan Resilience.
    Tidak ada stimulus.
    """
    random.seed(seed)
    state = GraceState()
    rows = []

    story_effect = {
        "N": 3,
        "A": 1,
    }

    for day in range(1, days + 1):
        event = random.choice(EVENTS)

        state = apply_effect(state, event["effect"])
        state = apply_effect(state, story_effect)

        score = calculate_grace_index(state)
        zone = get_zone(score)

        rows.append({
            "day": day,
            "condition": "Story Only",
            "event": event["name"],
            "stimulus": "a_none",
            "N": round(state.N, 2),
            "S": round(state.S, 2),
            "R": round(state.R, 2),
            "C": round(state.C, 2),
            "E": round(state.E, 2),
            "A": round(state.A, 2),
            "GRACE": round(score, 2),
            "zone": zone,
        })

    return pd.DataFrame(rows)


def run_story_stimulus(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Story + Stimulus:
    Event memengaruhi state.
    Story effect memperkuat N dan A.
    Rule-based smart stimulus diterapkan.
    """
    random.seed(seed)
    state = GraceState()
    rows = []

    story_effect = {
        "N": 3,
        "A": 1,
    }

    for day in range(1, days + 1):
        event = random.choice(EVENTS)

        state = apply_effect(state, event["effect"])
        state = apply_effect(state, story_effect)

        stimulus = choose_stimulus(state, event["recommended"])
        state = apply_effect(state, STIMULUS_EFFECTS[stimulus])

        score = calculate_grace_index(state)
        zone = get_zone(score)

        rows.append({
            "day": day,
            "condition": "Story + Stimulus",
            "event": event["name"],
            "stimulus": stimulus,
            "N": round(state.N, 2),
            "S": round(state.S, 2),
            "R": round(state.R, 2),
            "C": round(state.C, 2),
            "E": round(state.E, 2),
            "A": round(state.A, 2),
            "GRACE": round(score, 2),
            "zone": zone,
        })

    return pd.DataFrame(rows)


def run_experiment_suite(
    profile: dict,
    days: int = 30,
    seed: int = 42,
    include_llm: bool = False,
    ollama_url: str = "http://100.112.221.112:11434",
    model: str = "gemma4",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan semua kondisi eksperimen.
    Return:
    - all_daily_df: data harian semua kondisi
    - summary_df: ringkasan metrik per kondisi
    """
    all_runs = []
    summaries = []

    baseline_df = run_baseline(days=days, seed=seed)
    all_runs.append(baseline_df)
    summaries.append(summarize_result(baseline_df, "Baseline"))

    story_only_df = run_story_only(days=days, seed=seed)
    all_runs.append(story_only_df)
    summaries.append(summarize_result(story_only_df, "Story Only"))

    story_stimulus_df = run_story_stimulus(days=days, seed=seed)
    all_runs.append(story_stimulus_df)
    summaries.append(summarize_result(story_stimulus_df, "Story + Stimulus"))

    if include_llm:
        llm_df = run_simulation_with_llm(
            profile=profile,
            days=days,
            seed=seed,
            ollama_url=ollama_url,
            model=model,
        )
        llm_df["condition"] = "Full GRACE LLM"
        all_runs.append(llm_df)
        summaries.append(summarize_result(llm_df, "Full GRACE LLM"))

    all_daily_df = pd.concat(all_runs, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    os.makedirs("data", exist_ok=True)
    all_daily_df.to_csv("data/experiment_daily_results.csv", index=False)
    summary_df.to_csv("data/experiment_summary.csv", index=False)

    return all_daily_df, summary_df