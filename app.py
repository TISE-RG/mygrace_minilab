import os
import pandas as pd
import streamlit as st
from story_agent import generate_simple_story, detect_values, recommend_prompt
from simulator import run_simulation

os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="MyGRACE MiniLab", layout="wide")

st.title("MyGRACE MiniLab")
st.caption("Quick win: Story Companion + Digital Twin Simulator")

page = st.sidebar.radio(
    "Pilih halaman",
    ["Story Companion", "Digital Twin Simulator"]
)

if page == "Story Companion":
    st.header("Story Companion")

    elder_name = st.text_input("Nama lansia", value="Ibu Maria")
    story_type = st.selectbox(
        "Jenis cerita",
        ["Hari ini", "Masa lalu", "Masa depan"]
    )

    raw_story = st.text_area(
        "Ceritakan dengan sederhana",
        height=180,
        placeholder="Contoh: Hari ini saya duduk di teras. Anak belum menelepon, tetapi tetangga menyapa saya."
    )

    if st.button("Buat Narasi Bermakna"):
        if not raw_story.strip():
            st.warning("Tuliskan cerita terlebih dahulu.")
        else:
            narrative = generate_simple_story(raw_story, story_type)
            values = detect_values(raw_story)
            prompt = recommend_prompt(values)

            st.subheader("Narasi Bermakna")
            st.write(narrative)

            st.subheader("Nilai yang Terlihat")
            st.write(", ".join(values))

            st.subheader("Saran Lanjutan")
            st.info(prompt)

            row = {
                "name": elder_name,
                "story_type": story_type,
                "raw_story": raw_story,
                "narrative": narrative,
                "values": ", ".join(values),
                "prompt": prompt
            }

            path = "data/stories.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])

            df.to_csv(path, index=False)
            st.success("Cerita disimpan.")

    if os.path.exists("data/stories.csv"):
        st.subheader("Cerita Tersimpan")
        st.dataframe(pd.read_csv("data/stories.csv"))

if page == "Digital Twin Simulator":
    st.header("Digital Twin Simulator")

    days = st.slider("Jumlah hari simulasi", 7, 90, 30)
    seed = st.number_input("Seed simulasi", value=42)

    if st.button("Jalankan Simulasi"):
        df = run_simulation(days=days, seed=int(seed))

        st.subheader("Grafik GRACE Index")
        st.line_chart(df.set_index("day")["GRACE"])

        st.subheader("Tabel Simulasi Harian")
        st.dataframe(df)

        stable_days = len(df[df["GRACE"] >= 70])
        vulnerable_days = len(df[df["GRACE"] < 70])

        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata GRACE", round(df["GRACE"].mean(), 2))
        col2.metric("Stable Days", stable_days)
        col3.metric("Vulnerable/Disrupted Days", vulnerable_days)

        df.to_csv("data/simulation_results.csv", index=False)
        st.success("Hasil simulasi disimpan.")