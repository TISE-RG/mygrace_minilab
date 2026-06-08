import os
import pandas as pd
import streamlit as st

from story_agent import (
    generate_simple_story,
    analyze_grace_with_llm,
    get_ollama_models,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OLLAMA_MODEL,
)
from grace_core import GraceState, calculate_grace_index, get_zone
from simulator import run_simulation
from elder_profiles import ELDER_PROFILES
from simulator_llm import run_simulation_with_llm


# =========================
# Setup
# =========================

os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="MyGRACE MiniLab",
    layout="wide"
)

st.title("MyGRACE MiniLab")
st.caption(
    "Quick win Tahap 2: Story Companion + Digital Twin Simulator "
    "+ Ollama LLM + LLM Daily Story"
)


# =========================
# Sidebar: Navigation & Ollama Settings
# =========================

st.sidebar.title("Navigasi")

page = st.sidebar.radio(
    "Pilih halaman",
    ["Story Companion", "Digital Twin Simulator"]
)

st.sidebar.divider()

st.sidebar.subheader("Ollama Settings")

ollama_url = st.sidebar.text_input(
    "Alamat Server Ollama",
    value=st.session_state.get("ollama_url", DEFAULT_OLLAMA_URL),
    help="Contoh: http://100.112.221.112:11434"
)

st.session_state["ollama_url"] = ollama_url

if "available_models" not in st.session_state:
    st.session_state["available_models"] = []

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = DEFAULT_OLLAMA_MODEL

col_test, col_refresh = st.sidebar.columns(2)

with col_test:
    test_clicked = st.button("Test")

with col_refresh:
    refresh_clicked = st.button("Refresh Model")

if test_clicked or refresh_clicked:
    models = get_ollama_models(ollama_url)

    if models:
        st.session_state["available_models"] = models

        if st.session_state["selected_model"] not in models:
            st.session_state["selected_model"] = models[0]

        st.sidebar.success(f"Ollama terhubung. {len(models)} model ditemukan.")
    else:
        st.sidebar.error("Gagal mengambil model. Periksa alamat Ollama.")

available_models = st.session_state.get("available_models", [])

if available_models:
    selected_model = st.sidebar.selectbox(
        "Pilih Model",
        options=available_models,
        index=(
            available_models.index(st.session_state["selected_model"])
            if st.session_state["selected_model"] in available_models
            else 0
        )
    )
else:
    selected_model = st.sidebar.text_input(
        "Nama Model",
        value=st.session_state.get("selected_model", DEFAULT_OLLAMA_MODEL),
        help="Isi manual bila daftar model belum berhasil diambil."
    )

st.session_state["selected_model"] = selected_model

st.sidebar.caption(f"Server aktif: `{ollama_url}`")
st.sidebar.caption(f"Model aktif: `{selected_model}`")


# =========================
# Page 1: Story Companion
# =========================

if page == "Story Companion":
    st.header("Story Companion")
    st.write(
        "Halaman ini membantu lansia menuliskan cerita hidup, "
        "lalu GRACE membantu merapikan, memaknai, dan memberi analisis awal."
    )

    with st.expander("Pengaturan LLM aktif", expanded=False):
        st.write("Ollama URL:", ollama_url)
        st.write("Model:", selected_model)

    elder_name = st.text_input(
        "Nama lansia",
        value="Ibu Maria"
    )

    story_type = st.selectbox(
        "Jenis cerita",
        ["Hari ini", "Masa lalu", "Masa depan"]
    )

    raw_story = st.text_area(
        "Ceritakan dengan sederhana",
        height=180,
        placeholder=(
            "Contoh: Hari ini saya duduk di teras. "
            "Anak belum menelepon, tetapi tetangga menyapa saya."
        )
    )

    if st.button("Buat Narasi Bermakna"):
        if not raw_story.strip():
            st.warning("Tuliskan cerita terlebih dahulu.")
        else:
            with st.spinner("GRACE sedang membaca dan memaknai cerita..."):
                narrative = generate_simple_story(
                    raw_story=raw_story,
                    story_type=story_type,
                    elder_name=elder_name,
                    ollama_url=ollama_url,
                    model=selected_model,
                )

                grace_data = analyze_grace_with_llm(
                    raw_story=raw_story,
                    narrative=narrative,
                    ollama_url=ollama_url,
                    model=selected_model,
                )

                state = GraceState(
                    N=grace_data["N"],
                    S=grace_data["S"],
                    R=grace_data["R"],
                    C=grace_data["C"],
                    E=grace_data["E"],
                    A=grace_data["A"],
                )

                grace_score = calculate_grace_index(state)
                zone = get_zone(grace_score)

            st.subheader("Narasi Bermakna")
            st.write(narrative)

            st.subheader("GRACE Index")

            col1, col2 = st.columns(2)
            col1.metric("GRACE Score", round(grace_score, 2))
            col2.metric("Zone", zone)

            st.write(
                {
                    "Narrative Continuity": grace_data["N"],
                    "Spiritual Grounding": grace_data["S"],
                    "Relational Vitality": grace_data["R"],
                    "Contribution and Usefulness": grace_data["C"],
                    "Community and World Engagement": grace_data["E"],
                    "Adaptive Resilience": grace_data["A"],
                }
            )

            st.subheader("Nilai dan Emosi")
            st.write("Nilai:", ", ".join(grace_data.get("values", [])))
            st.write("Emosi:", ", ".join(grace_data.get("emotion", [])))

            st.subheader("Stimulus yang Disarankan")
            st.info(grace_data.get("recommended_stimulus", "naratif"))

            st.subheader("Catatan Risiko")
            risk_notes = grace_data.get("risk_notes", "Tidak ada catatan khusus.")
            if risk_notes and risk_notes.lower() != "tidak ada catatan khusus":
                st.warning(risk_notes)
            else:
                st.success("Tidak ada catatan khusus.")

            row = {
                "name": elder_name,
                "story_type": story_type,
                "raw_story": raw_story,
                "narrative": narrative,
                "N": grace_data["N"],
                "S": grace_data["S"],
                "R": grace_data["R"],
                "C": grace_data["C"],
                "E": grace_data["E"],
                "A": grace_data["A"],
                "GRACE": round(grace_score, 2),
                "zone": zone,
                "values": ", ".join(grace_data.get("values", [])),
                "emotion": ", ".join(grace_data.get("emotion", [])),
                "recommended_stimulus": grace_data.get(
                    "recommended_stimulus",
                    "naratif"
                ),
                "risk_notes": grace_data.get("risk_notes", ""),
                "ollama_url": ollama_url,
                "model": selected_model,
            }

            path = "data/stories.csv"

            if os.path.exists(path):
                df = pd.read_csv(path)
                df = pd.concat(
                    [df, pd.DataFrame([row])],
                    ignore_index=True
                )
            else:
                df = pd.DataFrame([row])

            df.to_csv(path, index=False)

            st.success("Cerita dan analisis GRACE disimpan.")

    st.divider()

    if os.path.exists("data/stories.csv"):
        st.subheader("Cerita Tersimpan")
        stories_df = pd.read_csv("data/stories.csv")
        st.dataframe(stories_df, width="stretch")
    else:
        st.info("Belum ada cerita tersimpan.")


# =========================
# Page 2: Digital Twin Simulator
# =========================

if page == "Digital Twin Simulator":
    st.header("Digital Twin Simulator")
    st.write(
        "Halaman ini menjalankan simulasi digital twin lansia. "
        "Setiap hari digital twin mengalami kejadian, GRACE Index berubah, "
        "dan sistem memilih stimulus yang sesuai."
    )

    with st.expander("Pengaturan LLM aktif", expanded=False):
        st.write("Ollama URL:", ollama_url)
        st.write("Model:", selected_model)

    profile_name = st.selectbox(
        "Pilih profil digital twin",
        list(ELDER_PROFILES.keys())
    )

    selected_profile = ELDER_PROFILES[profile_name]

    with st.expander("Lihat profil digital twin", expanded=False):
        st.json(selected_profile)

    col_a, col_b = st.columns(2)

    with col_a:
        days = st.slider(
            "Jumlah hari simulasi",
            min_value=7,
            max_value=90,
            value=30
        )

    with col_b:
        seed = st.number_input(
            "Seed simulasi",
            value=42,
            step=1
        )

    st.info(
        "Catatan: simulasi LLM memanggil model minimal dua kali per hari. "
        "Untuk awal, gunakan 7 hari dulu agar cepat."
    )

    col_run1, col_run2 = st.columns(2)

    with col_run1:
        run_rule_based = st.button("Jalankan Simulasi Rule-Based")

    with col_run2:
        run_llm_based = st.button("Jalankan Simulasi dengan LLM")

    if run_rule_based:
        df = run_simulation(
            days=days,
            seed=int(seed)
        )

        st.subheader("Grafik GRACE Index - Rule-Based")
        st.line_chart(df.set_index("day")["GRACE"])

        st.subheader("Tabel Simulasi Harian - Rule-Based")
        st.dataframe(df, width="stretch")

        stable_days = len(df[df["GRACE"] >= 70])
        vulnerable_days = len(df[df["GRACE"] < 70])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rata-rata GRACE",
            round(df["GRACE"].mean(), 2)
        )

        col2.metric(
            "Stable Days",
            stable_days
        )

        col3.metric(
            "Vulnerable/Disrupted Days",
            vulnerable_days
        )

        df.to_csv(
            "data/simulation_results_rule_based.csv",
            index=False
        )

        st.success(
            "Hasil simulasi rule-based disimpan di "
            "data/simulation_results_rule_based.csv."
        )

    if run_llm_based:
        with st.spinner("Menjalankan simulasi LLM. Ini bisa memakan waktu..."):
            df = run_simulation_with_llm(
                profile=selected_profile,
                days=days,
                seed=int(seed),
                ollama_url=ollama_url,
                model=selected_model,
            )

        st.subheader("Grafik GRACE Index - LLM")
        st.line_chart(df.set_index("day")["GRACE"])

        st.subheader("Tabel Simulasi Harian dengan Daily Story")
        st.dataframe(df, width="stretch")

        st.subheader("Contoh Daily Story")

        selected_day = st.slider(
            "Pilih hari untuk melihat cerita",
            min_value=1,
            max_value=len(df),
            value=1
        )

        story = df.loc[df["day"] == selected_day, "daily_story"].iloc[0]
        event = df.loc[df["day"] == selected_day, "event"].iloc[0]
        stimulus = df.loc[df["day"] == selected_day, "stimulus"].iloc[0]
        zone = df.loc[df["day"] == selected_day, "zone"].iloc[0]
        grace_score = df.loc[df["day"] == selected_day, "GRACE"].iloc[0]

        st.write(f"**Hari:** {selected_day}")
        st.write(f"**Event:** {event}")
        st.write(f"**Stimulus:** {stimulus}")
        st.write(f"**GRACE Score:** {grace_score}")
        st.write(f"**Zone:** {zone}")
        st.write(story)

        stable_days = len(df[df["GRACE"] >= 70])
        vulnerable_days = len(df[df["GRACE"] < 70])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rata-rata GRACE",
            round(df["GRACE"].mean(), 2)
        )

        col2.metric(
            "Stable Days",
            stable_days
        )

        col3.metric(
            "Vulnerable/Disrupted Days",
            vulnerable_days
        )

        df.to_csv(
            "data/simulation_results_llm.csv",
            index=False
        )

        st.success(
            "Hasil simulasi LLM disimpan di "
            "data/simulation_results_llm.csv."
        )

    st.divider()

    st.subheader("Hasil Simulasi Tersimpan")

    col_saved1, col_saved2 = st.columns(2)

    with col_saved1:
        if os.path.exists("data/simulation_results_rule_based.csv"):
            st.write("Rule-Based terakhir")
            sim_rule_df = pd.read_csv("data/simulation_results_rule_based.csv")
            st.dataframe(sim_rule_df, width="stretch")
        else:
            st.info("Belum ada hasil simulasi rule-based.")

    with col_saved2:
        if os.path.exists("data/simulation_results_llm.csv"):
            st.write("LLM terakhir")
            sim_llm_df = pd.read_csv("data/simulation_results_llm.csv")
            st.dataframe(sim_llm_df, width="stretch")
        else:
            st.info("Belum ada hasil simulasi LLM.")
