import os
import json
import pandas as pd
import streamlit as st

from story_agent import (
    generate_simple_story,
    analyze_grace_with_llm,
    get_ollama_models,
    benchmark_ollama,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OLLAMA_MODEL,
)
from grace_core import GraceState, calculate_grace_index, get_zone
from simulator import run_simulation
from elder_profiles import ELDER_PROFILES
from simulator_llm import run_simulation_with_llm
from experiment_runner import run_experiment_suite


# =========================
# Helpers
# =========================

def load_all_story_sources() -> pd.DataFrame:
    frames = []

    if os.path.exists("data/stories.csv"):
        web_df = pd.read_csv("data/stories.csv")
        web_df["source"] = "web_app"
        frames.append(web_df)

    if os.path.exists("data/telegram_stories.csv"):
        tg_df = pd.read_csv("data/telegram_stories.csv")
        tg_df["source"] = "telegram"
        frames.append(tg_df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    if "timestamp" not in df.columns:
        df["timestamp"] = ""

    return df


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
    "Story Companion + Digital Twin Simulator + Experiment Dashboard "
    "+ Telegram Data Entry + Ollama Benchmark"
)


# =========================
# Sidebar: Navigation & Ollama Settings
# =========================

st.sidebar.title("Navigasi")

page = st.sidebar.radio(
    "Pilih halaman",
    ["Story Companion", "Digital Twin Simulator", "Experiment Dashboard"]
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
# Sidebar: Ollama Benchmark
# =========================

st.sidebar.divider()
st.sidebar.subheader("Ollama Speed Test")

benchmark_prompt = st.sidebar.text_area(
    "Prompt benchmark",
    value=(
        "Tuliskan satu paragraf pendek dalam bahasa Indonesia tentang "
        "seorang lansia yang menemukan makna dari hari yang sederhana."
    ),
    height=100
)

if st.sidebar.button("Benchmark Ollama"):
    with st.sidebar.spinner("Mengukur kecepatan Ollama..."):
        bench = benchmark_ollama(
            ollama_url=ollama_url,
            model=selected_model,
            prompt=benchmark_prompt
        )

    if bench.get("ok"):
        benchmark_row = {
            "server": bench["server"],
            "model": bench["model"],
            "wall_time_sec": bench["wall_time_sec"],
            "total_duration_sec": bench["total_duration_sec"],
            "load_duration_sec": bench["load_duration_sec"],
            "prompt_eval_count": bench["prompt_eval_count"],
            "prompt_tokens_per_sec": bench["prompt_tokens_per_sec"],
            "eval_count": bench["eval_count"],
            "output_tokens_per_sec": bench["output_tokens_per_sec"],
        }

        benchmark_path = "data/ollama_benchmark.csv"

        if os.path.exists(benchmark_path):
            benchmark_df = pd.read_csv(benchmark_path)
            benchmark_df = pd.concat(
                [benchmark_df, pd.DataFrame([benchmark_row])],
                ignore_index=True
            )
        else:
            benchmark_df = pd.DataFrame([benchmark_row])

        benchmark_df.to_csv(benchmark_path, index=False)

        st.sidebar.success("Benchmark selesai.")

        st.sidebar.metric(
            "Output tokens/sec",
            bench["output_tokens_per_sec"]
        )

        st.sidebar.metric(
            "Prompt tokens/sec",
            bench["prompt_tokens_per_sec"]
        )

        st.sidebar.metric(
            "Wall time/sec",
            bench["wall_time_sec"]
        )

        with st.sidebar.expander("Detail benchmark", expanded=False):
            st.json({
                "server": bench["server"],
                "model": bench["model"],
                "wall_time_sec": bench["wall_time_sec"],
                "total_duration_sec": bench["total_duration_sec"],
                "load_duration_sec": bench["load_duration_sec"],
                "prompt_eval_count": bench["prompt_eval_count"],
                "prompt_eval_duration_sec": bench["prompt_eval_duration_sec"],
                "prompt_tokens_per_sec": bench["prompt_tokens_per_sec"],
                "eval_count": bench["eval_count"],
                "eval_duration_sec": bench["eval_duration_sec"],
                "output_tokens_per_sec": bench["output_tokens_per_sec"],
            })

        with st.sidebar.expander("Preview respons", expanded=False):
            st.write(bench["response_preview"])

    else:
        st.sidebar.error(f"Benchmark gagal: {bench.get('error')}")

if os.path.exists("data/ollama_benchmark.csv"):
    with st.sidebar.expander("Riwayat benchmark", expanded=False):
        st.dataframe(
            pd.read_csv("data/ollama_benchmark.csv"),
            use_container_width=True
        )


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

            st.subheader("Sumber Skoring")
            st.write(grace_data.get("scoring_source", "unknown"))

            with st.expander("Alasan Skoring", expanded=False):
                st.json(grace_data.get("scoring_reason", {}))

            with st.expander("Raw LLM Scoring Response", expanded=False):
                st.write(grace_data.get("raw_llm_scoring_response", ""))

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
                "scoring_source": grace_data.get("scoring_source", ""),
                "scoring_reason": json.dumps(
                    grace_data.get("scoring_reason", {}),
                    ensure_ascii=False
                ) if "json" in globals() else str(grace_data.get("scoring_reason", {})),
                "raw_llm_scoring_response": grace_data.get("raw_llm_scoring_response", ""),
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
        st.subheader("Cerita Tersimpan dari Web App")
        stories_df = pd.read_csv("data/stories.csv")
        st.dataframe(stories_df, use_container_width=True)
    else:
        st.info("Belum ada cerita tersimpan dari Web App.")

    st.divider()

    if os.path.exists("data/telegram_stories.csv"):
        st.subheader("Cerita Masuk dari Telegram")
        telegram_df = pd.read_csv("data/telegram_stories.csv")
        st.dataframe(telegram_df, use_container_width=True)
    else:
        st.info("Belum ada cerita dari Telegram.")

    st.divider()
    st.subheader("Integrated Story History")

    all_story_df = load_all_story_sources()

    if not all_story_df.empty:
        st.dataframe(all_story_df, use_container_width=True)

        if "GRACE" in all_story_df.columns:
            numeric_df = all_story_df.copy()
            numeric_df["GRACE"] = pd.to_numeric(numeric_df["GRACE"], errors="coerce")
            numeric_df = numeric_df.dropna(subset=["GRACE"])

            if not numeric_df.empty:
                col1, col2, col3 = st.columns(3)

                col1.metric("Total entries", len(numeric_df))
                col2.metric("Average GRACE", round(numeric_df["GRACE"].mean(), 2))
                col3.metric(
                    "GRACE range",
                    f"{numeric_df['GRACE'].min()}–{numeric_df['GRACE'].max()}"
                )

                st.subheader("GRACE by Entry")
                chart_df = numeric_df.reset_index()
                chart_df["entry"] = chart_df.index + 1
                st.line_chart(chart_df.set_index("entry")["GRACE"])

                if "source" in numeric_df.columns:
                    st.subheader("Average GRACE by Source")
                    source_summary = (
                        numeric_df.groupby("source")["GRACE"]
                        .mean()
                        .reset_index()
                    )
                    st.bar_chart(source_summary.set_index("source"))

                if "recommended_stimulus" in numeric_df.columns:
                    st.subheader("Stimulus Distribution")
                    stim_summary = (
                        numeric_df["recommended_stimulus"]
                        .fillna("unknown")
                        .value_counts()
                        .reset_index()
                    )
                    stim_summary.columns = ["stimulus", "count"]
                    st.bar_chart(stim_summary.set_index("stimulus"))

                if "zone" in numeric_df.columns:
                    st.subheader("Zone Distribution")
                    zone_summary = (
                        numeric_df["zone"]
                        .fillna("unknown")
                        .value_counts()
                        .reset_index()
                    )
                    zone_summary.columns = ["zone", "count"]
                    st.bar_chart(zone_summary.set_index("zone"))

                if "scoring_source" in numeric_df.columns:
                    st.subheader("Scoring Source Distribution")
                    scoring_summary = (
                        numeric_df["scoring_source"]
                        .fillna("unknown")
                        .value_counts()
                        .reset_index()
                    )
                    scoring_summary.columns = ["scoring_source", "count"]
                    st.bar_chart(scoring_summary.set_index("scoring_source"))
    else:
        st.info("Belum ada cerita dari sumber mana pun.")


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
        st.dataframe(df, use_container_width=True)

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
        st.dataframe(df, use_container_width=True)

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
            st.dataframe(sim_rule_df, use_container_width=True)
        else:
            st.info("Belum ada hasil simulasi rule-based.")

    with col_saved2:
        if os.path.exists("data/simulation_results_llm.csv"):
            st.write("LLM terakhir")
            sim_llm_df = pd.read_csv("data/simulation_results_llm.csv")
            st.dataframe(sim_llm_df, use_container_width=True)
        else:
            st.info("Belum ada hasil simulasi LLM.")


# =========================
# Page 3: Experiment Dashboard
# =========================

if page == "Experiment Dashboard":
    st.header("Experiment Dashboard")
    st.write(
        "Halaman ini membandingkan beberapa kondisi eksperimen: "
        "Baseline, Story Only, Story + Stimulus, dan Full GRACE LLM."
    )

    with st.expander("Pengaturan LLM aktif", expanded=False):
        st.write("Ollama URL:", ollama_url)
        st.write("Model:", selected_model)

    profile_name = st.selectbox(
        "Pilih profil digital twin untuk eksperimen",
        list(ELDER_PROFILES.keys()),
        key="experiment_profile"
    )

    selected_profile = ELDER_PROFILES[profile_name]

    with st.expander("Lihat profil digital twin", expanded=False):
        st.json(selected_profile)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        exp_days = st.slider(
            "Jumlah hari eksperimen",
            min_value=7,
            max_value=90,
            value=30,
            key="exp_days"
        )

    with col_b:
        exp_seed = st.number_input(
            "Seed eksperimen",
            value=42,
            step=1,
            key="exp_seed"
        )

    with col_c:
        include_llm = st.checkbox(
            "Sertakan Full GRACE LLM",
            value=False,
            help=(
                "Jika dicentang, eksperimen akan memanggil LLM berkali-kali. "
                "Gunakan 7 hari dulu untuk tes awal."
            )
        )

    st.info(
        "Saran: jalankan dulu tanpa LLM untuk melihat perbandingan cepat. "
        "Setelah itu centang Full GRACE LLM dengan 7 hari."
    )

    if st.button("Jalankan Experiment Suite"):
        with st.spinner("Menjalankan eksperimen..."):
            all_daily_df, summary_df = run_experiment_suite(
                profile=selected_profile,
                days=exp_days,
                seed=int(exp_seed),
                include_llm=include_llm,
                ollama_url=ollama_url,
                model=selected_model,
            )

        st.success("Eksperimen selesai.")

        st.subheader("Ringkasan Hasil Eksperimen")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Perbandingan Average GRACE")
        avg_chart_df = summary_df.set_index("condition")[["average_grace"]]
        st.bar_chart(avg_chart_df)

        st.subheader("Perbandingan Stable Days")
        stable_chart_df = summary_df.set_index("condition")[["stable_days"]]
        st.bar_chart(stable_chart_df)

        st.subheader("Perbandingan Vulnerable Days")
        vuln_chart_df = summary_df.set_index("condition")[["vulnerable_days"]]
        st.bar_chart(vuln_chart_df)

        st.subheader("Kurva GRACE per Hari")

        pivot_df = all_daily_df.pivot_table(
            index="day",
            columns="condition",
            values="GRACE",
            aggfunc="mean"
        )

        st.line_chart(pivot_df)

        st.subheader("Data Harian Semua Kondisi")
        st.dataframe(all_daily_df, use_container_width=True)

        st.download_button(
            label="Download Ringkasan Eksperimen CSV",
            data=summary_df.to_csv(index=False),
            file_name="experiment_summary.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Data Harian CSV",
            data=all_daily_df.to_csv(index=False),
            file_name="experiment_daily_results.csv",
            mime="text/csv"
        )

    st.divider()

    st.subheader("Hasil Eksperimen Tersimpan")

    if os.path.exists("data/experiment_summary.csv"):
        saved_summary = pd.read_csv("data/experiment_summary.csv")
        st.write("Ringkasan terakhir")
        st.dataframe(saved_summary, use_container_width=True)
    else:
        st.info("Belum ada ringkasan eksperimen tersimpan.")

    if os.path.exists("data/experiment_daily_results.csv"):
        saved_daily = pd.read_csv("data/experiment_daily_results.csv")
        st.write("Data harian terakhir")
        st.dataframe(saved_daily, use_container_width=True)
    else:
        st.info("Belum ada data harian eksperimen tersimpan.")